/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.aika.training;


import org.aika.neuron.Synapse;
import org.aika.Model;
import org.aika.Neuron;
import org.aika.corpus.Conflicts;
import org.aika.corpus.Document;
import org.aika.corpus.InterpretationNode;
import org.aika.lattice.NodeActivation;
import org.aika.neuron.Activation;
import org.aika.neuron.INeuron;
import org.aika.neuron.Synapse;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.TreeSet;

import static org.aika.corpus.InterpretationNode.State.SELECTED;
import static org.aika.corpus.InterpretationNode.State.EXCLUDED;


/**
 * Meta-neurons and meta-synapses allow to generate new fully trained neurons based on a single training data set.
 * The meta-network employs an inhibitory feedback loop to determine if a certain information is already known (i.e.
 * there is already a neuron representing it) or if it is new knowledge that should be represented by a new neuron. If
 * there is already a neuron that represents this information, then the meta-neuron will be suppressed by the
 * feedback loop. Otherwise the meta-neuron will be activated which means a copy of the meta-neuron will be generated
 * using only the meta information of this neuron. The meta-synapses are only added to this new neuron if the input
 * neuron has beed active in the training data set as well. If the input neuron of the meta-synapse is another
 * inhibitory neuron, then the resulting synapse of the new neuron is going to be connected to the activated input
 * neuron of this inhibitory neuron.
 *
 * @author Lukas Molzberger
 */
public class MetaNetwork {


    public static void train(Document doc) {
        long v = doc.visitedCounter++;
        for(INeuron n: new ArrayList<>(doc.finallyActivatedNeurons)) {
            if(n.type == INeuron.Type.INHIBITORY) {
                for (Activation sAct : n.getFinalActivations(doc)) {
                    for(Activation.SynapseActivation sa: sAct.getFinalInputActivations()) {
                        Activation act = sa.input;
                        Neuron targetNeuron = act.key.node.neuron;
                        boolean newNeuron = false;
                        if(targetNeuron.get().type == INeuron.Type.META) {
                            newNeuron = true;
                            targetNeuron = doc.model.createNeuron(n.label.substring(2) + "-" + doc.getText(act.key.range));
                            INeuron.update(doc.model, doc.threadId, targetNeuron, n.bias, Collections.emptySet());
                        }

                        Activation metaNeuronAct = getMetaNeuronAct(sAct);
                        if(metaNeuronAct != null && metaNeuronAct.visited != v) {
                            metaNeuronAct.visited = v;
                            transferMetaSynapses(doc, metaNeuronAct, targetNeuron, newNeuron, n.provider, v);
                        }
                    }
                }
            }
        }
    }


    private static Activation getMetaNeuronAct(Activation sAct) {
        for(Activation.SynapseActivation sa: sAct.neuronInputs) {
            if(sa.input.key.node.neuron.get().label.startsWith("M-")) {
                return sa.input;
            }
        }
        return null;
    }


    private static void transferMetaSynapses(Document doc, Activation metaAct, Neuron targetNeuron, boolean newNeuron, Neuron supprN, long v) {
        TreeSet<Synapse> inputSynapses = new TreeSet<>(Synapse.INPUT_SYNAPSE_COMP);

        Integer ridOffset = computeRidOffset(metaAct);
        for (Activation.SynapseActivation sa : metaAct.getFinalInputActivations()) {
            MetaSynapse ss = sa.synapse.meta;
            if (ss != null && (ss.metaWeight != 0.0 || ss.metaBias != 0.0)) {
                Neuron ina = sa.input.key.node.neuron;
                Neuron inb = null;
                Integer rid = null;

                if (ina.get().type == INeuron.Type.INHIBITORY && ss.metaWeight >= 0.0) {
                    List<Activation.SynapseActivation> inputs = sa.input.getFinalInputActivations();
                    for(Activation.SynapseActivation iSA: inputs) {
                        Activation iAct = iSA.input;
                        inb = iAct.key.node.neuron;
                        rid = iAct.key.rid;
                    }
                } else {
                    inb = ina;
                    rid = sa.input.key.rid;
                }

                if (inb != null) {
                    Synapse.Key osk = sa.synapse.key;
                    Synapse.Key nsk = new Synapse.Key(
                            osk.isRecurrent,
                            osk.relativeRid != null ?
                                    osk.relativeRid :
                                    (ss.metaRelativeRid && ridOffset != null && rid != null ? rid - ridOffset : null),
                            osk.absoluteRid,
                            osk.rangeMatch,
                            osk.rangeOutput
                    );

                    Synapse ns = new Synapse(inb, targetNeuron, nsk);
                    if (!ns.exists()) {
                        ns.weightDelta = ss.metaWeight;
                        ns.biasDelta = ss.metaBias;

                        inputSynapses.add(ns);
                    }
                }
            }
        }

        INeuron.update(doc.model, doc.threadId, targetNeuron, newNeuron ? metaAct.key.node.neuron.get().metaBias : 0.0, inputSynapses);

        if (newNeuron) {
            Activation.SynapseActivation inhibMetaLink = metaAct.getFinalOutputActivations().get(0);
            Synapse.Key inhibSynKey = inhibMetaLink.synapse.key;
            MetaSynapse inhibSS = inhibMetaLink.synapse.meta;
            supprN.addSynapse(
                    new Synapse.Builder()
                            .setNeuron(targetNeuron)
                            .setWeight(inhibSS.metaWeight)
                            .setBias(inhibSS.metaBias)
                            .setRelativeRid(inhibSynKey.relativeRid)
                            .setAbsoluteRid(inhibSynKey.absoluteRid)
                            .setRangeMatch(inhibSynKey.rangeMatch)
                            .setRangeOutput(inhibSynKey.rangeOutput)
            );

            Activation.Key mak = metaAct.key;
            mak.interpretation.setState(EXCLUDED, v);
        }

        for(Synapse s: inputSynapses) {
            for (Activation iAct : s.input.get().getFinalActivations(doc)) {
                iAct.upperBound = 0.0;
                repropagate(doc, iAct);
            }
        }

        doc.propagate();

//        doc.process();

        for(Activation tAct: targetNeuron.get().getAllActivations(doc)) {
            if(!isConflicting(tAct, doc.visitedCounter++)) {
                tAct.key.interpretation.setState(SELECTED, v);

                Activation sAct = getOutputAct(tAct.neuronOutputs, INeuron.Type.INHIBITORY);
                sAct.key.interpretation.setState(SELECTED, v);

                Activation mAct = getOutputAct(sAct.neuronOutputs, INeuron.Type.META);

                ArrayList<Activation> newActs = new ArrayList<>();
                if (mAct != null) {
                    mAct.visited = v;
                    newActs.add(mAct);
                }
                newActs.add(tAct);
                newActs.add(sAct);

                newActs.forEach(act -> doc.vQueue.add(0, act));
                doc.vQueue.processChanges(doc.selectedSearchNode, doc.visitedCounter++, doc.selectedSearchNode.visited);

                if (tAct.getFinalState().value <= 0.0) {
                    tAct.key.interpretation.setState(EXCLUDED, v);
                    mAct.key.interpretation.setState(SELECTED, doc.selectedSearchNode.visited);

                    newActs.forEach(act -> doc.vQueue.add(0, act));
                    doc.vQueue.processChanges(doc.selectedSearchNode, doc.visitedCounter++, doc.selectedSearchNode.visited);
                }

                for (Activation act : newActs) {
                    if (act.isFinalActivation()) {
                        doc.finallyActivatedNeurons.add(act.key.node.neuron.get(doc));
                    }
                }
            }
        }
    }


    private static boolean isConflicting(Activation tAct, long v) {
        ArrayList<InterpretationNode> tmp = new ArrayList<>();
        Conflicts.collectConflicting(tmp, tAct.key.interpretation, v);
        for(InterpretationNode c: tmp) {
            if(c.state == SELECTED) return true;
        }
        return false;
    }


    private static Activation getOutputAct(TreeSet<Activation.SynapseActivation> outputActs, INeuron.Type type) {
        for(Activation.SynapseActivation sa: outputActs) {
            if(sa.output.key.node.neuron.get().type == type) {
                return sa.output;
            }
        }
        return null;
    }


    private static void repropagate(Document doc, NodeActivation<?> act) {
        act.key.node.propagateAddedActivation(doc, act);
        for(NodeActivation<?> oAct: act.outputs.values()) {
            if(!(oAct instanceof Activation)) {
                repropagate(doc, oAct);
            }
        }
    }


    private static Integer computeRidOffset(Activation mAct) {
        for (Activation.SynapseActivation sa : mAct.getFinalInputActivations()) {
            if(sa.synapse.key.relativeRid != null && sa.input.key.rid != null) {
                return sa.input.key.rid - sa.synapse.key.relativeRid;
            }
        }
        return null;
    }


    public static Neuron initMetaNeuron(Model m, Neuron n, double bias, double metaBias, Synapse.Builder... inputs) {
        n.get().metaBias = metaBias;
        return Neuron.init(n, bias, INeuron.Type.META, inputs);
    }
}
