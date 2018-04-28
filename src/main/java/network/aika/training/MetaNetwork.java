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
package network.aika.training;


import network.aika.Document;
import network.aika.Utils;
import network.aika.neuron.Neuron;
import network.aika.neuron.Synapse;
import network.aika.neuron.activation.Activation;
import network.aika.neuron.INeuron;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;


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

    private static final Logger log = LoggerFactory.getLogger(MetaNetwork.class);

    public static void train(Document doc) {
        Map<Activation, List<Target>> metaActivations = new TreeMap<>();

        List<INeuron> inhibitoryNeurons = doc.finallyActivatedNeurons
                .stream()
                .filter(n -> n.type == INeuron.Type.INHIBITORY)
                .collect(Collectors.toList());

        for (INeuron n : inhibitoryNeurons) {
            for (Activation inhibAct : n.getActivations(doc, true)) {
                for (Activation.SynapseActivation sa : inhibAct.getFinalInputActivations()) {
                    Activation act = sa.input;
                    Neuron targetNeuron = act.getNeuron();

                    doc.createV = doc.visitedCounter++;

                    boolean newNeuron = false;
                    if (targetNeuron.get().type == INeuron.Type.META) {
                        newNeuron = true;
                        targetNeuron = doc.model.createNeuron(n.label.substring(2) + "-" + doc.getText(act.range));
                        INeuron.update(doc.model, doc.threadId, doc, targetNeuron, n.bias, Collections.emptySet());
                    }

                    Activation metaNeuronAct = getMetaNeuronAct(inhibAct);
                    if (metaNeuronAct != null) {
                        List<Target> targets = metaActivations.get(metaNeuronAct);
                        if(targets == null) {
                            targets = new ArrayList<>();
                            metaActivations.put(metaNeuronAct, targets);
                        }

                        targets.add(new Target(targetNeuron, newNeuron, n.provider));
                    }
                }
            }
        }

        for(Map.Entry<Activation, List<Target>> me: metaActivations.entrySet()) {
            for(Target t: me.getValue()) {
                transferMetaSynapses(doc, metaActivations, me.getKey(), t);
            }
        }
    }


    private static class Target {
        Neuron targetNeuron;
        boolean isNewNeuron;
        Neuron inhibNeuron;

        public Target(Neuron targetNeuron, boolean isNewNeuron, Neuron inhibNeuron) {
            this.targetNeuron = targetNeuron;
            this.isNewNeuron = isNewNeuron;
            this.inhibNeuron = inhibNeuron;
        }
    }


    private static Activation getMetaNeuronAct(Activation inhibAct) {
        for(Activation.SynapseActivation sa: inhibAct.neuronInputs.values()) {
            if(sa.input.getINeuron().type == INeuron.Type.META) {
                return sa.input;
            }
        }
        return null;
    }


    private static void transferMetaSynapses(Document doc, Map<Activation, List<Target>> metaActivations, Activation metaAct, Target t) {
        TreeSet<Synapse> inputSynapses = new TreeSet<>(Synapse.INPUT_SYNAPSE_COMP);

        for (Activation.SynapseActivation sa : metaAct.getFinalInputActivations()) {
            MetaSynapse inputMetaSyanpse = sa.synapse.meta;
            Synapse os = sa.synapse;

            if (inputMetaSyanpse != null && (inputMetaSyanpse.metaWeight != 0.0 || inputMetaSyanpse.metaBias != 0.0)) {
                Neuron ina = sa.input.node.neuron;

                List<Activation.SynapseActivation> inputs = ina.get().type == INeuron.Type.INHIBITORY && inputMetaSyanpse.metaWeight >= 0.0 ?
                        sa.input.getFinalInputActivations() :
                        Collections.singletonList(sa);

                for(Activation.SynapseActivation isa: inputs) {
                    Neuron in = isa.input.getNeuron();

                    if(in.get(doc).type == INeuron.Type.META) {
                        List<Target> inputTargets = metaActivations.get(isa.input);
                        if(inputTargets != null) {
                            for (Target it : metaActivations.get(isa.input)) {
                                createOrLookupSynapse(doc, t, inputSynapses, inputMetaSyanpse, os, it.targetNeuron);
                            }
                        }
                    } else {
                        createOrLookupSynapse(doc, t, inputSynapses, inputMetaSyanpse, os, in);
                    }
                }
            }
        }

        if(log.isDebugEnabled()) {
            log.debug(showDelta(t.targetNeuron.get(), inputSynapses));
        }

        INeuron.update(doc.model, doc.threadId, doc, t.targetNeuron, t.isNewNeuron ? metaAct.getINeuron().metaBias : 0.0, inputSynapses);

        if (t.isNewNeuron) {
            Activation.SynapseActivation inhibMetaLink = metaAct.getFinalOutputActivations().get(0);
            Synapse.Key inhibSynKey = inhibMetaLink.synapse.key;
            MetaSynapse inhibSS = inhibMetaLink.synapse.meta;
            t.inhibNeuron.addSynapse(
                    new Synapse.Builder()
                            .setNeuron(t.targetNeuron)
                            .setWeight(inhibSS.metaWeight)
                            .setBias(inhibSS.metaBias)
                            .addRelations(inhibMetaLink.synapse.relations)
                            .setRangeOutput(inhibSynKey.rangeOutput)
            );
        }

        doc.propagate();
    }


    private static void createOrLookupSynapse(Document doc, Target t, TreeSet<Synapse> inputSynapses, MetaSynapse inputMetaSyanpse, Synapse os, Neuron in) {
        Synapse ns = new Synapse(in, t.targetNeuron, os.id, os.key);
        if (!ns.exists()) {
            ns.updateDelta(doc, inputMetaSyanpse.metaWeight, inputMetaSyanpse.metaBias);

            inputSynapses.add(ns);
        }
    }


    public static String showDelta(INeuron n, Set<Synapse> synapses) {
        StringBuilder sb = new StringBuilder();

        sb.append("N: " + n.label + " ob:" + n.biasSum + " nb:" + (n.biasSum + n.biasSumDelta) + "\n");
        for (Synapse s : synapses) {
            if (s.weightDelta != 0.0) {
                Integer f = null;
                if (s.input.get().statistic != null) {
                    f = ((NeuronStatistic) s.input.get().statistic).frequency;
                }

                sb.append("    S:" + s.input.getLabel() + " ow:" + s.weight + " nw:" + s.getNewWeight() + (f != null ? " f:" + f : "") + " " + (s.isConjunction(false, false) ? "CONJ" : "DISJ") + "\n");
            }
        }
        return sb.toString();
    }


    public static Neuron initMetaNeuron(Neuron n, double bias, double metaBias, Synapse.Builder... inputs) {
        n.get().metaBias = metaBias;
        return Neuron.init(n, bias, INeuron.Type.META, inputs);
    }
}