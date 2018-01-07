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
package org.aika.neuron;


import org.aika.*;
import org.aika.neuron.Activation.State;
import org.aika.neuron.Activation.SynapseActivation;
import org.aika.corpus.*;
import org.aika.corpus.SearchNode.Coverage;
import org.aika.lattice.InputNode;
import org.aika.lattice.Node;
import org.aika.lattice.NodeActivation;
import org.aika.lattice.OrNode;
import org.aika.neuron.Synapse.Key;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.aika.lattice.Node.BEGIN_COMP;
import static org.aika.lattice.Node.END_COMP;
import static org.aika.lattice.Node.RID_COMP;
import static org.aika.neuron.Activation.State.*;

/**
 * The {@code INeuron} class represents a internal neuron implementation in Aikas neural network and is connected to other neurons through
 * input synapses and output synapses. The activation value of a neuron is calculated by computing the weighted sum
 * (input act. value * synapse weight) of the input synapses, adding the bias to it and sending the resulting value
 * through a transfer function (the upper part of tanh).
 * <p>
 * <p>The neuron does not store its activations by itself. The activation objects are stored within the
 * logic nodes. To access the activations of this neuron simply use the member variable {@code node} or use
 * the method {@code getFinalActivations(Document doc)} to ge the final activations of this neuron.
 *
 * @author Lukas Molzberger
 */
public class INeuron extends AbstractNode<Neuron> implements Comparable<INeuron> {

    public static boolean ALLOW_WEAK_NEGATIVE_WEIGHTS = false;

    private static final Logger log = LoggerFactory.getLogger(INeuron.class);

    public static double WEIGHT_TOLERANCE = 0.001;
    public static double TOLERANCE = 0.000001;
    public static int MAX_SELF_REFERENCING_DEPTH = 5;

    public String label;
    public Type type;

    public enum Type {
        EXCITATORY,
        INHIBITORY,
        META
    }

    public String outputText;

    public volatile double bias;
    public volatile double biasDelta;
    public volatile double biasSum;
    public volatile double biasSumDelta;

    public volatile double metaBias = 0.0;


    public volatile double posDirSum;
    public volatile double negDirSum;
    public volatile double negRecSum;
    public volatile double posRecSum;

    public volatile double maxRecurrentSum = 0.0;

    public Writable statistic;

    public ActivationFunction activationFunction = ActivationFunction.RECTIFIED_SCALED_LOGISTIC_SIGMOID;
    public String activationFunctionKey = ActivationFunction.RECTIFIED_SCALED_LOGISTIC_SIGMOID_KEY;


    // A synapse is stored only in one direction, depending on the synapse weight.
    public TreeMap<Synapse, Synapse> inputSynapses = new TreeMap<>(Synapse.INPUT_SYNAPSE_COMP);
    public TreeMap<Synapse, Synapse> outputSynapses = new TreeMap<>(Synapse.OUTPUT_SYNAPSE_COMP);

    public TreeMap<Key, Provider<InputNode>> outputNodes = new TreeMap<>();

    public Provider<OrNode> node;


    public ReadWriteLock lock = new ReadWriteLock();


    public ThreadState[] threads;

    /**
     * The {@code ThreadState} is a thread local data structure containing the activations of a single document for
     * a specific logic node.
     */
    public static class ThreadState {
        public long lastUsed;

        public TreeMap<Activation.Key, Activation> activations;
        public TreeMap<Activation.Key, Activation> activationsEnd;
        public TreeMap<Activation.Key, Activation> activationsRid;

        public ThreadState() {
            activations = new TreeMap<>(BEGIN_COMP);
            activationsEnd = new TreeMap<>(END_COMP);
            activationsRid = new TreeMap<>(RID_COMP);
        }
    }


    public ThreadState getThreadState(int threadId, boolean create) {
        ThreadState th = threads[threadId];
        if (th == null) {
            if (!create) return null;

            th = new ThreadState();
            threads[threadId] = th;
        }
        th.lastUsed = provider.model.docIdCounter.get();
        return th;
    }



    private INeuron() {
    }


    public INeuron(Model m) {
        this(m, null);
    }


    public INeuron(Model m, String label) {
        this(m, label, null);
    }


    public INeuron(Model m, String label, String outputText) {
        this.label = label;
        this.outputText = outputText;

        if(m.neuronStatisticFactory != null) {
            statistic = m.neuronStatisticFactory.createStatisticObject();
        }

        threads = new ThreadState[m.numberOfThreads];

        provider = new Neuron(m, this);

        OrNode node = new OrNode(m);

        node.neuron = provider;
        this.node = node.provider;

        setModified();
    }

    /**
     * Propagate an input activation into the network.
     *
     * @param doc   The current document
     * @param begin The range begin
     * @param end   The range end
     * @param rid   The relational id (e.g. the word position)
     * @param o     The interpretation node
     * @param value The activation value of this input activation
     */
    public Activation addInput(Document doc, int begin, int end, Integer rid, InterprNode o, double value, Double targetValue) {
        Node.addActivationAndPropagate(doc, new NodeActivation.Key(node.get(doc), new Range(begin, end), rid, o), Collections.emptySet());

        doc.propagate();

        Activation act = Activation.get(doc, this, rid, new Range(begin, end), Range.Relation.EQUALS, o, InterprNode.Relation.EQUALS);
        State s = new State(value, 0, NormWeight.ZERO_WEIGHT);
        act.rounds.set(0, s);
        act.inputValue = value;
        act.targetValue = targetValue;

        doc.inputNeuronActivations.add(act);
        doc.finallyActivatedNeurons.add(act.key.node.neuron.get(doc));

        doc.ubQueue.add(act);

        if(targetValue != null) {
            doc.targetActivations.add(act);
        }

        doc.propagate();

        return act;
    }


    // TODO
    public void remove() {

        clearActivations();

        for (Synapse s : inputSynapses.values()) {
            INeuron in = s.input.get();
            in.provider.lock.acquireWriteLock();
            in.provider.inMemoryOutputSynapses.remove(s);
            in.provider.lock.releaseWriteLock();
        }

        provider.lock.acquireReadLock();
        for (Synapse s : provider.inMemoryOutputSynapses.values()) {
            INeuron out = s.output.get();
            out.lock.acquireWriteLock();
            out.inputSynapses.remove(s);
            out.lock.releaseWriteLock();
        }
        provider.lock.releaseReadLock();
    }


    public void propagateAddedActivation(Document doc, Activation act) {
        doc.ubQueue.add(act);
    }


    public void computeBounds(Activation act) {
        double ub = biasSum + posRecSum;
        double lb = biasSum + posRecSum;

        for (SynapseActivation sa : act.neuronInputs) {
            Synapse s = sa.synapse;
            Activation iAct = sa.input;

            if (iAct == act) continue;

            if (s.isNegative()) {
                if (!checkSelfReferencing(act.key.interpretation, iAct.key.interpretation, 0) && act.key.interpretation.contains(iAct.key.interpretation, true)) {
                    ub += iAct.lowerBound * s.weight;
                }

                lb += s.weight;
            } else {
                ub += iAct.upperBound * s.weight;
                lb += iAct.lowerBound * s.weight;
            }
        }

        act.upperBound = activationFunction.f(ub);
        act.lowerBound = activationFunction.f(lb);
    }


    public State computeWeight(int round, Activation act, SearchNode sn) {
        Coverage c = sn.getCoverage(act.key.interpretation);
        if(c == Coverage.UNKNOWN) return State.ZERO;

        double[] sum = {biasSum, 0.0};

        int fired = -1;

        for (InputState is: getInputStates(act, round, sn)) {
            Synapse s = is.sa.synapse;
            Activation iAct = is.sa.input;

            if (iAct == act) continue;

            int t = is.sa.isRecurrent() ? REC : DIR;
            sum[t] += is.s.value * s.weight;

            if (!is.sa.isRecurrent() && !s.isNegative() && sum[DIR] + sum[REC] >= 0.0 && fired < 0) {
                fired = iAct.rounds.get(round).fired + 1;
            }
        }

        double drSum = sum[DIR] + sum[REC];
        double currentActValue = activationFunction.f(drSum);

        act.maxActValue = Math.max(act.maxActValue, currentActValue);

        // Compute only the recurrent part is above the threshold.
        NormWeight newWeight = NormWeight.create(
                c == Coverage.SELECTED ? (sum[DIR] + negRecSum) < 0.0 ? Math.max(0.0, drSum) : sum[REC] - negRecSum : 0.0,
                (sum[DIR] + negRecSum) < 0.0 ? Math.max(0.0, sum[DIR] + negRecSum + maxRecurrentSum) : maxRecurrentSum
        );

        return new State(
                c == Coverage.SELECTED || ALLOW_WEAK_NEGATIVE_WEIGHTS ? currentActValue : 0.0,
                c == Coverage.SELECTED || ALLOW_WEAK_NEGATIVE_WEIGHTS ? fired : -1,
                newWeight
        );
    }


    private List<InputState> getInputStates(Activation act, int round, SearchNode sn) {
        InterprNode o = act.key.interpretation;
        ArrayList<InputState> tmp = new ArrayList<>();
        Synapse lastSynapse = null;
        InputState maxInputState = null;
        for (SynapseActivation sa : act.neuronInputs) {
            if (lastSynapse != null && lastSynapse != sa.synapse) {
                tmp.add(maxInputState);
                maxInputState = null;
            }

            State s = getInputState(round, sn, o, sa);
            if (maxInputState == null || maxInputState.s.value < s.value) {
                maxInputState = new InputState(sa, s);
            }
            lastSynapse = sa.synapse;
        }
        if (maxInputState != null) {
            tmp.add(maxInputState);
        }

        return tmp;
    }


    private State getInputState(int round, SearchNode sn, InterprNode o, SynapseActivation sa) {
        Activation iAct = sa.input;
        InterprNode io = iAct.key.interpretation;

        State is = State.ZERO;
        if (sa.isRecurrent()) {
            if (!sa.synapse.isNegative() || !checkSelfReferencingForSelected(o, io, 0)) {
                is = round == 0 ? getInitialState(sn.getCoverage(io)) : iAct.rounds.get(round - 1);
            }
        } else {
            is = iAct.rounds.get(round);
        }
        return is;
    }


    private State getInitialState(Coverage c) {
        return new State(
                c == Coverage.SELECTED ? 1.0 : 0.0,
                0,
                NormWeight.ZERO_WEIGHT
        );
    }


    private static class InputState {
        public InputState(SynapseActivation sa, State s) {
            this.sa = sa;
            this.s = s;
        }

        SynapseActivation sa;
        State s;
    }


    public void computeOutputErrorSignal(Activation act) {
        if(act.targetValue != null) {
            act.errorSignal += act.targetValue - act.getFinalState().value;
        }

        act.updateErrorSignal();
    }


    public void computeBackpropagationErrorSignal(Activation act) {
        for (SynapseActivation sa : act.neuronOutputs) {
            Synapse s = sa.synapse;
            Activation oAct = sa.output;

            act.errorSignal += s.weight * oAct.errorSignal * (1.0 - act.getFinalState().value);
        }

        act.updateErrorSignal();
    }


    public void train(Document doc, Activation targetAct, double learnRate, Document.SynapseEvaluation se) {
        if (Math.abs(targetAct.errorSignal) < TOLERANCE) return;

        long v = doc.visitedCounter++;

        double x = learnRate * targetAct.errorSignal;
        biasDelta += x;
        for (INeuron n : doc.finallyActivatedNeurons) {
            for(Activation iAct: n.getFinalActivations(doc)) {
                Document.SynEvalResult ser = se.evaluate(iAct, targetAct);
                if(ser != null) {
                    trainSynapse(doc, iAct, ser, x, v);
                }
            }
        }

        doc.notifyWeightsModified(this, provider.inMemoryInputSynapses.values());
    }


    private void trainSynapse(Document doc, Activation iAct, Document.SynEvalResult ser, double x, long v) {
        if (iAct.visited == v) return;
        iAct.visited = v;

        INeuron inputNeuron = iAct.key.node.neuron.get(doc);
        if(inputNeuron == this) {
            return;
        }
        double deltaW = x * ser.significance * iAct.getFinalState().value;

        Provider<InputNode> inp = inputNeuron.outputNodes.get(ser.synapseKey.createInputNodeKey());
        Synapse synapse = null;
        InputNode in = null;
        if(inp != null) {
            in = inp.get(doc);
            synapse = in.getSynapse(ser.synapseKey.relativeRid, provider);
        }

        if(synapse == null) {
            synapse = new Synapse(inputNeuron.provider, provider, ser.synapseKey);

            if(in == null) {
                in = InputNode.add(provider.model, ser.synapseKey.createInputNodeKey(), synapse.input.get(doc));
            }
            in.setSynapse(synapse);
            synapse.link();
        }

        synapse.weightDelta = (float) deltaW;
    }


    private static boolean checkSelfReferencing(InterprNode nx, InterprNode ny, int depth) {
        if (nx == ny) return true;

        if (depth > MAX_SELF_REFERENCING_DEPTH) return false;

        if (ny.orInterprNodes != null) {
            for (InterprNode n : ny.orInterprNodes) {
                if (checkSelfReferencing(nx, n, depth + 1)) return true;
            }
        }

        return false;
    }


    private static boolean checkSelfReferencingForSelected(InterprNode nx, InterprNode ny, int depth) {
        if (nx == ny || nx.contains(ny, true)) return true;

        if (depth > MAX_SELF_REFERENCING_DEPTH) return false;

        if (ny.selectedOrInterprNodes != null) {
            for (InterprNode n : ny.selectedOrInterprNodes) {
                if (checkSelfReferencingForSelected(nx, n, depth + 1)) return true;
            }
        }

        return false;
    }


    /**
     * Sets the incoming and outgoing links between neuron activations.
     *
     * @param doc
     * @param act
     */
    public void linkNeuronRelations(Document doc, Activation act) {
        long v = doc.visitedCounter++;
        lock.acquireReadLock();
        linkNeuronActs(doc, act, v, 0);
        linkNeuronActs(doc, act, v, 1);
        lock.releaseReadLock();
    }


    private void linkNeuronActs(Document doc, Activation act, long v, int dir) {
        ArrayList<Activation> recNegTmp = new ArrayList<>();

        provider.lock.acquireReadLock();
        NavigableMap<Synapse, Synapse> syns = (dir == 0 ? provider.inMemoryInputSynapses : provider.inMemoryOutputSynapses);

        for (Synapse s : getActiveSynapses(provider.model, doc, dir, syns)) {
            Neuron p = (dir == 0 ? s.input : s.output);
            INeuron an = p.getIfNotSuspended();
            if (an != null) {
                ThreadState th = an.getThreadState(doc.threadId, false);
                if (th == null || th.activations.isEmpty()) continue;

                linkActSyn(an, doc, act, dir, recNegTmp, s);
            }
        }
        provider.lock.releaseReadLock();

        for (Activation rAct : recNegTmp) {
            Activation oAct = (dir == 0 ? act : rAct);
            Activation iAct = (dir == 0 ? rAct : act);

            markConflicts(iAct, oAct, v);

            addConflict(doc, oAct.key.interpretation, iAct.key.interpretation, iAct, Collections.singleton(act), v);
        }
    }


    private static void addConflict(Document doc, InterprNode io, InterprNode o, NodeActivation act, Collection<NodeActivation> inputActs, long v) {
        if (o.markedConflict == v || o.fixed == Boolean.TRUE) {
            if (!isAllowed(doc, io, o, inputActs)) {
                Conflicts.add(act, io, o);
            }
        } else {
            if(o.orInterprNodes != null) {
                for (InterprNode no : o.orInterprNodes) {
                    addConflict(doc, io, no, act, inputActs, v);
                }
            }
        }
    }


    private static boolean isAllowed(Document doc, InterprNode io, InterprNode o, Collection<NodeActivation> inputActs) {
        if (io != null && o.contains(io, false)) return true;
        for (NodeActivation act : inputActs) {
            if (act.key.node.isAllowedOption(doc.threadId, o, act, doc.visitedCounter++)) return true;
        }
        return false;
    }


    private static void markConflicts(Activation iAct, Activation oAct, long v) {
        oAct.key.interpretation.markedConflict = v;
        for (SynapseActivation sa : iAct.neuronOutputs) {
            if (sa.isRecurrent() && sa.synapse.isNegative()) {
                sa.output.key.interpretation.markedConflict = v;
            }
        }
    }


    private static void linkActSyn(INeuron n, Document doc, Activation act, int dir, ArrayList<Activation> recNegTmp, Synapse s) {
        Synapse.Key sk = s.key;

        Integer rid;
        if (dir == 0) {
            rid = sk.absoluteRid != null ? sk.absoluteRid : Utils.nullSafeAdd(act.key.rid, false, sk.relativeRid, false);
        } else {
            rid = Utils.nullSafeSub(act.key.rid, false, sk.relativeRid, false);
        }

        Stream<Activation> tmp = Activation.select(
                doc,
                n,
                rid,
                act.key.range,
                dir == 0 ? sk.rangeMatch.invert() : sk.rangeMatch,
                null,
                null
        );

        final int d = dir;
        tmp.forEach(rAct -> {
            Activation oAct = (d == 0 ? act : rAct);
            Activation iAct = (d == 0 ? rAct : act);

            SynapseActivation sa = new SynapseActivation(s, iAct, oAct);
            iAct.addSynapseActivation(0, sa);
            oAct.addSynapseActivation(1, sa);

            if (s.isNegative()) {
                recNegTmp.add(rAct);
            }
        });
    }


    private static Collection<Synapse> getActiveSynapses(Model m, Document doc, int dir, NavigableMap<Synapse, Synapse> syns) {
        // Optimization in case the set of synapses is very large
        if (syns.size() < 10 || doc.activatedNeurons.size() * 20 > syns.size()) {
            return syns.values();
        }

        Collection<Synapse> synsTmp;
        ArrayList<Synapse> newSyns = new ArrayList<>();
        Synapse lk = new Synapse(null, null, Synapse.Key.MIN_KEY);
        Synapse uk = new Synapse(null, null, Synapse.Key.MAX_KEY);

        for (INeuron n : doc.activatedNeurons) {
            if (dir == 0) {
                lk.input = n.provider;
                uk.input = n.provider;
            } else {
                lk.output = n.provider;
                uk.output = n.provider;
            }

            // Using addAll is not efficient here.
            for (Synapse s : syns.subMap(lk, true, uk, true).values()) {
                newSyns.add(s);
            }
        }

        synsTmp = newSyns;
        return synsTmp;
    }



    public Collection<Activation> getActivations(Document doc) {
        ThreadState th = getThreadState(doc.threadId, false);
        if (th == null) return Collections.EMPTY_LIST;
        return th.activations.values();
    }


    public synchronized Activation getFirstActivation(Document doc) {
        ThreadState th = getThreadState(doc.threadId, false);
        if (th == null || th.activations.isEmpty()) return null;
        return th.activations.firstEntry().getValue();
    }



    public void clearActivations() {
        for (int i = 0; i < provider.model.numberOfThreads; i++) {
            clearActivations(i);
        }
    }


    public void clearActivations(Document doc) {
        clearActivations(doc.threadId);
    }


    public void clearActivations(int threadId) {
        ThreadState th = getThreadState(threadId, false);
        if (th == null) return;
        th.activations.clear();

        if (th.activationsEnd != null) th.activationsEnd.clear();
        if (th.activationsRid != null) th.activationsRid.clear();
    }


    public int compareTo(INeuron n) {
        if (provider.id < n.provider.id) return -1;
        else if (provider.id > n.provider.id) return 1;
        else return 0;
    }


    @Override
    public void write(DataOutput out) throws IOException {
        out.writeBoolean(true);

        out.writeBoolean(label != null);
        if(label != null) {
            out.writeUTF(label);
        }

        out.writeBoolean(type != null);
        if(type != null) {
            out.writeUTF(type.name());
        }

        out.writeBoolean(outputText != null);
        if(outputText != null) {
            out.writeUTF(outputText);
        }

        out.writeBoolean(statistic != null);
        if(statistic != null) {
            statistic.write(out);
        }

        out.writeDouble(bias);
        out.writeDouble(biasSum);
        out.writeDouble(posDirSum);
        out.writeDouble(negDirSum);
        out.writeDouble(negRecSum);
        out.writeDouble(posRecSum);
        out.writeDouble(maxRecurrentSum);

        out.writeUTF(activationFunctionKey);

        out.writeInt(outputNodes.size());
        for (Map.Entry<Key, Provider<InputNode>> me : outputNodes.entrySet()) {
            me.getKey().write(out);
            out.writeInt(me.getValue().id);
        }

        out.writeBoolean(node != null);
        if (node != null) {
            out.writeInt(node.id);
        }

        for (Synapse s : inputSynapses.values()) {
            if (s.input != null) {
                out.writeBoolean(true);
                s.write(out);
            }
        }
        out.writeBoolean(false);
        for (Synapse s : outputSynapses.values()) {
            if (s.output != null) {
                out.writeBoolean(true);
                s.write(out);
            }
        }
        out.writeBoolean(false);
    }


    @Override
    public void readFields(DataInput in, Model m) throws IOException {
        if(in.readBoolean()) {
            label = in.readUTF();
        }

        if(in.readBoolean()) {
            type = Type.valueOf(in.readUTF());
        }

        if(in.readBoolean()) {
            outputText = in.readUTF();
        }

        if(in.readBoolean() && m.neuronStatisticFactory != null) {
            statistic = m.neuronStatisticFactory.createStatisticObject();
            statistic.readFields(in, m);
        }

        bias = in.readDouble();
        biasSum = in.readDouble();
        posDirSum = in.readDouble();
        negDirSum = in.readDouble();
        negRecSum = in.readDouble();
        posRecSum = in.readDouble();
        maxRecurrentSum = in.readDouble();

        activationFunctionKey = in.readUTF();
        activationFunction = m.activationFunctions.get(activationFunctionKey);

        int s = in.readInt();
        for (int i = 0; i < s; i++) {
            Key k = Key.read(in, m);
            Provider<InputNode> n = m.lookupNodeProvider(in.readInt());
            outputNodes.put(k, n);
        }

        if (in.readBoolean()) {
            Integer nId = in.readInt();
            node = m.lookupNodeProvider(nId);
        }

        while (in.readBoolean()) {
            Synapse syn = Synapse.read(in, m);

            inputSynapses.put(syn, syn);
        }

        while (in.readBoolean()) {
            Synapse syn = Synapse.read(in, m);

            outputSynapses.put(syn, syn);
        }
    }


    @Override
    public void suspend() {
        for (Synapse s : inputSynapses.values()) {
            s.input.removeInMemoryOutputSynapse(s);
        }
        for (Synapse s : outputSynapses.values()) {
            s.output.removeInMemoryInputSynapse(s);
        }

        provider.lock.acquireReadLock();
        for (Synapse s : provider.inMemoryInputSynapses.values()) {
            if(!s.isConjunction) {
                s.input.removeInMemoryOutputSynapse(s);
            }
        }
        for (Synapse s : provider.inMemoryOutputSynapses.values()) {
            if(s.isConjunction) {
                s.output.removeInMemoryInputSynapse(s);
            }
        }
        provider.lock.releaseReadLock();
    }


    @Override
    public void reactivate() {
        provider.lock.acquireReadLock();
        for (Synapse s : provider.inMemoryInputSynapses.values()) {
            if(!s.isConjunction) {
                s.input.addInMemoryOutputSynapse(s);
            }
        }
        for (Synapse s : provider.inMemoryOutputSynapses.values()) {
            if(s.isConjunction) {
                s.output.addInMemoryInputSynapse(s);
            }
        }
        provider.lock.releaseReadLock();

        for (Synapse s : inputSynapses.values()) {
            s.input.addInMemoryOutputSynapse(s);
            if (!s.input.isSuspended()) {
                s.output.addInMemoryInputSynapse(s);
            }
        }
        for (Synapse s : outputSynapses.values()) {
            s.output.addInMemoryInputSynapse(s);
            if (!s.output.isSuspended()) {
                s.input.addInMemoryOutputSynapse(s);
            }
        }
    }


    public static Neuron update(Model m, int threadId, Neuron pn, double biasDelta, Collection<Synapse> modifiedSynapses) {
        INeuron n = pn.get();
        n.biasDelta += biasDelta;
        n.biasSumDelta += biasDelta;

        modifiedSynapses.forEach(s -> n.biasSumDelta += s.biasDelta);
        // s.link requires an updated n.biasSumDelta value.
        modifiedSynapses.forEach(s -> s.link());

        if (!Converter.convert(m, threadId, n, modifiedSynapses)) return null;

        return n.provider;
    }


    public static INeuron readNeuron(DataInput in, Neuron p) throws IOException {
        INeuron n = new INeuron();
        n.provider = p;
        n.threads = new ThreadState[p.model.numberOfThreads];
        n.readFields(in, p.model);
        return n;
    }


    public String toString() {
        return label;
    }


    public String toStringWithSynapses() {
        SortedSet<Synapse> is = new TreeSet<>(new Comparator<Synapse>() {
            @Override
            public int compare(Synapse s1, Synapse s2) {
                int r = Double.compare(s2.weight, s1.weight);
                if (r != 0) return r;
                return Integer.compare(s1.input.id, s2.input.id);
            }
        });

        is.addAll(inputSynapses.values());

        StringBuilder sb = new StringBuilder();
        sb.append(toString());
        sb.append("<");
        sb.append("B:");
        sb.append(Utils.round(biasSum));
        for (Synapse s : is) {
            sb.append(", ");
            sb.append(Utils.round(s.weight));
            sb.append(":");
            sb.append(s.key.relativeRid);
            sb.append(":");
            sb.append(s.input.toString());
        }
        sb.append(">");
        return sb.toString();
    }


    /**
     * {@code getFinalActivations} is a convenience method to retrieve all activations of the given neuron that
     * are part of the final interpretation. Before calling this method, the {@code doc.process()} needs to
     * be called first.
     *
     * @param doc The current document
     * @return A collection with all final activations of this neuron.
     */
    public Collection<Activation> getFinalActivations(Document doc) {
        Stream<Activation> s = Activation.select(doc, this, null, null, null, null, null);
        return s.filter(act -> act.isFinalActivation())
                .collect(Collectors.toList());
    }


    public Collection<Activation> getAllActivations(Document doc) {
        Stream<Activation> s = Activation.select(doc, this, null, null, null, null, null);
        return s.collect(Collectors.toList());
    }


    public static class NormWeight {
        public final static NormWeight ZERO_WEIGHT = new NormWeight(0.0, 0.0);

        public final double w;
        public final double n;

        private NormWeight(double w, double n) {
            this.w = w;
            this.n = n;
        }

        public static NormWeight create(double w, double n) {
            assert w >= 0.0 && n >= 0.0;
            if (w == 0.0 && n == 0.0) return ZERO_WEIGHT;
            return new NormWeight(w, n);
        }

        public NormWeight add(NormWeight nw) {
            if (nw == null || nw == ZERO_WEIGHT) return this;
            return new NormWeight(w + nw.w, n + nw.n);
        }

        public NormWeight sub(NormWeight nw) {
            if (nw == null || nw == ZERO_WEIGHT) return this;
            return new NormWeight(w - nw.w, n - nw.n);
        }

        public double getNormWeight() {
            return n > 0 ? w / n : 0.0;
        }


        public boolean equals(NormWeight nw) {
            return (Math.abs(w - nw.w) <= INeuron.WEIGHT_TOLERANCE && Math.abs(n - nw.n) <= INeuron.WEIGHT_TOLERANCE);
        }

        public String toString() {
            return "W:" + w + " N:" + n + " NW:" + getNormWeight();
        }
    }


}
