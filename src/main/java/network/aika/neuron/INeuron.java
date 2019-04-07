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
package network.aika.neuron;


import network.aika.*;
import network.aika.lattice.Node;
import network.aika.lattice.NodeActivation;
import network.aika.neuron.activation.Activation;
import network.aika.neuron.activation.Linker;
import network.aika.neuron.activation.Position;
import network.aika.lattice.InputNode;
import network.aika.neuron.relation.Relation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static network.aika.neuron.INeuron.Type.EXCITATORY;
import static network.aika.neuron.INeuron.Type.INPUT;
import static network.aika.neuron.Synapse.OUTPUT;
import static network.aika.neuron.Synapse.State.CURRENT;
import static network.aika.neuron.Synapse.State.NEXT;

/**
 * The {@code INeuron} class represents a internal neuron implementation in Aikas neural network and is connected to other neurons through
 * input synapses and output synapses. The activation value of a neuron is calculated by computing the weighted sum
 * (input act. value * synapse weight) of the input synapses, adding the bias to it and sending the resulting value
 * through a transfer function (the upper part of tanh).
 * <p>
 * <p>The neuron does not store its activationsBySlotAndPosition by itself. The activation objects are stored within the
 * logic nodes. To access the activationsBySlotAndPosition of this neuron simply use the member variable {@code node} or use
 * the method {@code getFinalActivations(Document doc)} to ge the final activationsBySlotAndPosition of this neuron.
 *
 * @author Lukas Molzberger
 */
public class INeuron extends AbstractNode<Neuron> implements Comparable<INeuron> {

    private static final Logger log = LoggerFactory.getLogger(INeuron.class);

    public static boolean ALLOW_WEAK_NEGATIVE_WEIGHTS = false;
    public static double WEIGHT_TOLERANCE = 0.001;


    public static final INeuron MIN_NEURON = new INeuron();
    public static final INeuron MAX_NEURON = new INeuron();

    String label;
    Type type;


    public enum Type {
        INPUT,
        EXCITATORY,
        INHIBITORY
    }


    private String outputText;

    private volatile double bias;
    private volatile double biasDelta;

    private SynapseSummary synapseSummary = new SynapseSummary();

    private Writable extension;

    ActivationFunction activationFunction;


    private volatile int synapseIdCounter = 0;

    // synapseId -> relation
    private Map<Integer, Relation> outputRelations = new TreeMap<>();


    // A synapse is stored only in one direction, depending on the synapse weight.
    TreeMap<Synapse, Synapse> inputSynapses = new TreeMap<>(Synapse.INPUT_SYNAPSE_COMP);
    TreeMap<Synapse, Synapse> outputSynapses = new TreeMap<>(Synapse.OUTPUT_SYNAPSE_COMP);
    TreeMap<Synapse, Synapse> passiveInputSynapses = null;

    private Provider<InputNode> outputNode;
    private Node.OutputEntry inputNode;


    ReadWriteLock lock = new ReadWriteLock();


    PassiveInputFunction passiveInputFunction = null;


    private ThreadState[] threads;


    /**
     * The {@code ThreadState} is a thread local data structure containing the activationsBySlotAndPosition of a single document for
     * a specific logic node.
     */
    private static class ThreadState {
        public long lastUsed;

        private TreeMap<ActKey, Activation> activationsBySlotAndPosition;
        private TreeMap<Integer, Activation> activations;
        public int minLength = Integer.MAX_VALUE;
        public int maxLength = 0;


        public ThreadState() {
            activationsBySlotAndPosition = new TreeMap<>();
            activations = new TreeMap<>();
        }
    }


    public void setOutputNode(Provider<InputNode> node) {
        outputNode = node;
    }

    public Integer getId() {
        return provider.getId();
    }

    public String getLabel() {
        return label;
    }

    public Type getType() {
        return type;
    }


    public Provider<InputNode> getOutputNode() {
        return outputNode;
    }


    public Node.OutputEntry getInputNode() {
        return inputNode;
    }


    public void setInputNode(Node.OutputEntry n) {
        inputNode = n;
    }


    public SynapseSummary getSynapseSummary() {
        return synapseSummary;
    }


    public Map<Integer, Relation> getOutputRelations() {
        return outputRelations;
    }


    public Collection<Synapse> getInputSynapses() {
        return inputSynapses.values();
    }


    public Synapse getMaxInputSynapse(Synapse.State state) {
        if(type != EXCITATORY) {
            return null;
        }

        Synapse maxSyn = null;
        for(Synapse s: getInputSynapses()) {
            if(!s.isInactive()) {
                if(maxSyn == null || maxSyn.getNewWeight() < s.getNewWeight()) {
                    maxSyn = s;
                }
            }
        }
        return maxSyn;
    }


    public Collection<Synapse> getOutputSynapses() {
        return outputSynapses.values();
    }


    public Collection<Synapse> getPassiveInputSynapses() {
        if(passiveInputSynapses == null) {
             return Collections.emptyList();
        }

        return passiveInputSynapses.values();
    }


    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }


    public <T extends Writable> T getExtension() {
        return (T) extension;
    }


    public boolean addActivation(Activation act) {
        ThreadState th = getThreadState(act.getThreadId(), true);

        boolean first = th.activationsBySlotAndPosition.isEmpty();

        Integer l = act.length();
        if(l != null) {
            th.minLength = Math.min(th.minLength, l);
            th.maxLength = Math.max(th.maxLength, l);
        }

        for(Map.Entry<Integer, Position> me: act.getSlots().entrySet()) {
            ActKey ak = new ActKey(me.getKey(), me.getValue(), act.getId());
            th.activationsBySlotAndPosition.put(ak, act);
            th.activations.put(act.getId(), act);
        }

        return first;
    }


    public Stream<Activation> getActivations(Document doc) {
        ThreadState th = getThreadState(doc.getThreadId(), false);
        if(th == null) {
            return Stream.empty();
        }

        return th.activations.values().stream();
    }


    public boolean isEmpty(Document doc) {
        ThreadState th = getThreadState(doc.getThreadId(), false);
        if(th == null) {
            return true;
        }
        return th.activationsBySlotAndPosition.isEmpty();
    }


    public int size(Document doc) {
        ThreadState th = getThreadState(doc.getThreadId(), false);
        if(th == null) {
            return 0;
        }

        return th.activations.size();
    }


    public void clearActivations(Document doc) {
        ThreadState th = getThreadState(doc.getThreadId(), false);
        if(th == null) {
            return;
        }
        th.activationsBySlotAndPosition.clear();
        th.activations.clear();
    }


    public Stream<Activation> getActivations(Document doc, int slot, Position pos, boolean onlyFinal) {
        return getActivations(doc, slot, pos, true, slot, pos, false)
                .filter(act -> !onlyFinal || act.isFinalActivation());
    }


    public void clearActivations() {
        for (int i = 0; i < provider.getModel().numberOfThreads; i++) {
            clearActivations(i);
        }
    }


    public void clearActivations(int threadId) {
        ThreadState th = getThreadState(threadId, false);
        if (th == null) return;
        th.activationsBySlotAndPosition.clear();
        th.activations.clear();
    }


    public Stream<Activation> getActivations(Document doc, int fromSlot, Position fromPos, boolean fromInclusive, int toSlot, Position toPos, boolean toInclusive) {
        ThreadState th = getThreadState(doc.getThreadId(), false);
        if(th == null) {
            return Stream.empty();
        }
        return th.activationsBySlotAndPosition.subMap(
                new INeuron.ActKey(fromSlot, fromPos, Integer.MIN_VALUE),
                fromInclusive,
                new INeuron.ActKey(toSlot, toPos, Integer.MAX_VALUE),
                toInclusive
        ).values()
                .stream();
    }


    public Stream<Activation> getActivations(Document doc, boolean onlyFinal) {
        return onlyFinal ?
                getActivations(doc)
                        .filter(act -> act.isFinalActivation()) :
                getActivations(doc);
    }


    public Collection<Activation> getActivations(Document doc, SortedMap<Integer, Position> slots) {
        Integer firstSlot = slots.firstKey();
        Position firstPos = slots.get(firstSlot);

        return getActivations(doc, firstSlot, firstPos, true, firstSlot, firstPos, true)
                .filter( act -> {
                    for(Map.Entry<Integer, Position> me: slots.entrySet()) {
                        Position pos = me.getValue();
                        if(pos.getFinalPosition() != null && pos.compare(act.getSlot(me.getKey())) != 0) {
                            return false;
                        }
                    }
                    return true;
                })
                .collect(Collectors.toList());
    }


    public void unlinkInputSynapses() {
        for(Synapse s: inputSynapses.values()) {
            s.getInput().get().getOutputSynapses().remove(s);
        }
    }


    private static class ActKey implements Comparable<ActKey> {
        int slot;
        Position pos;
        int actId;

        public ActKey(int slot, Position pos, int actId) {
            this.slot = slot;
            this.pos = pos;
            this.actId = actId;
        }

        @Override
        public int compareTo(ActKey ak) {
            int r = Integer.compare(slot, ak.slot);
            if(r != 0) return r;
            r = pos.compare(ak.pos);
            if(r != 0) return r;
            return Integer.compare(actId, ak.actId);
        }
    }


    private ThreadState getThreadState(int threadId, boolean create) {
        ThreadState th = threads[threadId];
        if (th == null) {
            if (!create) return null;

            th = new ThreadState();
            threads[threadId] = th;
        }
        th.lastUsed = provider.getModel().docIdCounter.get();
        return th;
    }


    private INeuron() {
    }


    public INeuron(Model m, String label, String outputText, Type type, ActivationFunction actF) {
        this.label = label;
        this.type = type;
        this.activationFunction = actF;

        setOutputText(outputText);

        if(m.getNeuronExtensionFactory() != null) {
            extension = m.getNeuronExtensionFactory().createObject();
        }

        threads = new ThreadState[m.numberOfThreads];

        provider = new Neuron(m, this);

        InputNode iNode = new InputNode(m);

        iNode.setInputNeuron(provider);
        outputNode = iNode.getProvider();

        setModified();
        iNode.setModified();

    }


    public void setOutputText(String outputText) {
        this.outputText = outputText;
    }


    public String getOutputText() {
        return outputText;
    }

    /**
     * Propagate an input activation into the network.
     *
     * @param doc   The current document
     * @param input
     */
    public Activation addInput(Document doc, Activation.Builder input) {
        Activation act = getActivation(doc, input);

        if (act == null) {
            act = new Activation(doc, this, input.getSlots(doc));
        }

        act.setInputState(input);

        doc.addInputNeuronActivation(act);
        doc.addFinallyActivatedNeuron(act.getINeuron());

        if(getType() != INPUT) {
            doc.getLinker().linkInput(act);
            doc.getLinker().process();
        }

        propagate(act);

        doc.propagate();

        return act;
    }


    private Activation getActivation(Document doc, Activation.Builder input) {
        Integer firstSlot = input.positions.firstKey();
        Position firstPos = doc.lookupFinalPosition(input.positions.get(firstSlot));
        x: for(Activation a: getActivations(doc, firstSlot, firstPos, true, firstSlot, firstPos, true).collect(Collectors.toList())) {
            for(Map.Entry<Integer, Integer> me: input.positions.entrySet()) {
                Position pos = a.getSlot(me.getKey());
                if(pos == null || me.getValue().compareTo(pos.getFinalPosition()) != 0) {
                    continue x;
                }
            }
            return a;
        }
        return null;
    }


    public double getTotalBias(Synapse.State state) {
        switch(type) {
            case EXCITATORY:
                return getBias(state) - synapseSummary.getPosSum(state);
            case INHIBITORY:
                return getBias(state);
        }
        return getBias(state);
    }


    public void commit(Collection<Synapse> modifiedSynapses) {
        for (Synapse s : modifiedSynapses) {
            INeuron in = s.getInput().get();
            in.lock.acquireWriteLock();
            try {
                synapseSummary.updateSynapse(s);
            } finally {
                in.lock.releaseWriteLock();
            }
        }

        bias += biasDelta;
        biasDelta = 0.0;

        for (Synapse s : modifiedSynapses) {
            s.commit();

            if(s.isZero()) {
                s.unlink();
            }
        }

        synapseSummary.commit();

        setModified();
    }


    public Activation lookupActivation(Document doc, SortedMap<Integer, Position> slots, Predicate<Activation.Link> filter) {
        return getActivations(doc, slots)
                .stream()
                .filter(act -> act.match(filter))
                .findFirst()
                .orElse(null);
    }


    // TODO
    public void remove() {
        clearActivations();

        for (Synapse s : inputSynapses.values()) {
            INeuron in = s.getInput().get();
            in.provider.lock.acquireWriteLock();
            in.provider.activeOutputSynapses.remove(s);
            in.provider.lock.releaseWriteLock();
        }

        provider.lock.acquireReadLock();
        for (Synapse s : provider.activeOutputSynapses.values()) {
            INeuron out = s.getOutput().get();
            out.lock.acquireWriteLock();
            out.inputSynapses.remove(s);
            out.lock.releaseWriteLock();
        }
        provider.lock.releaseReadLock();
    }


    public synchronized int getNewSynapseId() {
        setModified();
        return synapseIdCounter++;
    }


    public synchronized void registerSynapseId(Integer synId) {
        if(synId >= synapseIdCounter) {
            setModified();
            synapseIdCounter = synId + 1;
        }
    }


    public void propagate(Activation act) {
        Document doc = act.getDocument();
        outputNode.get(doc).addActivation(act);

        if(outputSynapses != null) {
            for(Synapse s: outputSynapses.values()) {
                if(!s.isRecurrent() && (!s.isWeak(CURRENT) || s.getOutput().getType() == EXCITATORY)) {
                    propagate(s, act);
                }
            }
        }
    }


    private void propagate(Synapse s, Activation iAct) {
        Document doc = iAct.getDocument();
        INeuron n = s.getOutput().get(doc);

        Activation act = lookupActivation(s, iAct);
        /*Activation act = n.lookupActivation(doc, slots, l -> {
            Synapse s = l.getSynapse();
            if(!s.isIdentity()) return true;

            Integer i = oe.revSynapseIds.get(s.getId());
            Activation iAct = doc.getLinker().computeInputActivation(s, inputAct.getInputActivation(i));
            return i != null && l.getInput() == iAct;
        });
*/
        if(act == null) {
            act = new Activation(doc, n, getSlots(s, iAct));
        }

        doc.getUpperBoundQueue().add(act);

        Linker linker = doc.getLinker();
        linker.link(s, iAct, act);
        linker.process();
    }


    private SortedMap<Integer, Position> getSlots(Synapse s, Activation iAct) {
        SortedMap<Integer, Position> slots = new TreeMap<>();
        for (Map.Entry<Integer, Relation> me : s.getRelations().entrySet()) {
            Relation rel = me.getValue();
            if (me.getKey() == Synapse.OUTPUT) {
                rel.mapSlots(slots, iAct);
            }
        }
        return slots;
    }


    private Activation lookupActivation(Synapse os, Activation iAct) {
        for (Map.Entry<Integer, Relation> me : os.getRelations().entrySet()) {
            Integer relSynId = me.getKey();
            Relation rel = me.getValue();

            Activation existingAct = null;
            if (relSynId != OUTPUT) {
                Synapse rs = os.getOutput().getSynapseById(relSynId);
                if (rs != null) {
                    existingAct = rel
                            .invert()
                            .getActivations(rs.getInput().get(), iAct)
                            .flatMap(act -> act.getOutputLinksBySynapse(rs))
                            .map(rl -> rl.getOutput())
                            .findFirst()
                            .orElse(null);
                }
            } else {
                existingAct = rel
                        .invert()
                        .getActivations(os.getOutput().get(), iAct)
                        .findFirst()
                        .orElse(null);
            }

            if (existingAct != null) {
                return existingAct;
            }
        }

        return null;
    }


    public int compareTo(INeuron n) {
        if (this == n) return 0;
        if (this == MIN_NEURON) return -1;
        if (n == MIN_NEURON) return 1;
        if (this == MAX_NEURON) return 1;
        if (n == MAX_NEURON) return -1;

        if (getId() < n.getId()) return -1;
        else if (getId() > n.getId()) return 1;
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

        out.writeBoolean(extension != null);
        if(extension != null) {
            extension.write(out);
        }

        out.writeDouble(bias);

        synapseSummary.write(out);

        out.writeUTF(activationFunction.name());

        out.writeInt(outputNode.getId());

        out.writeBoolean(inputNode != null);
        if (inputNode != null) {
            inputNode.write(out);
        }

        out.writeInt(synapseIdCounter);
        for (Synapse s : inputSynapses.values()) {
            if (s.getInput() != null) {
                out.writeBoolean(true);
                s.write(out);

                out.writeBoolean(passiveInputSynapses != null && passiveInputSynapses.containsKey(s));
            }
        }
        out.writeBoolean(false);
        for (Synapse s : outputSynapses.values()) {
            if (s.getOutput() != null) {
                out.writeBoolean(true);
                s.write(out);
            }
        }
        out.writeBoolean(false);

        if(outputRelations != null) {
            out.writeInt(outputRelations.size());
            for (Map.Entry<Integer, Relation> me : outputRelations.entrySet()) {
                out.writeInt(me.getKey());

                me.getValue().write(out);
            }
        } else  {
            out.writeInt(0);
        }
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

        if(in.readBoolean()) {
            extension = m.getNeuronExtensionFactory().createObject();
            extension.readFields(in, m);
        }

        bias = in.readDouble();
        synapseSummary = SynapseSummary.read(in, m);

        activationFunction = ActivationFunction.valueOf(in.readUTF());

        outputNode = m.lookupNodeProvider(in.readInt());

        if (in.readBoolean()) {
            inputNode = Node.OutputEntry.read(in, m);
        }

        synapseIdCounter = in.readInt();
        while (in.readBoolean()) {
            Synapse syn = Synapse.read(in, m);
            inputSynapses.put(syn, syn);

            if(in.readBoolean()) {
                registerPassiveInputSynapse(syn);
            }
        }

        while (in.readBoolean()) {
            Synapse syn = Synapse.read(in, m);
            outputSynapses.put(syn, syn);
        }

        int l = in.readInt();
        if(l > 0) {
            outputRelations = new TreeMap<>();
            for(int i = 0; i < l; i++) {
                Integer relId = in.readInt();

                Relation r = Relation.read(in, m);
                outputRelations.put(relId, r);
            }
        }

        passiveInputFunction = m.passiveActivationFunctions.get(getId());
    }


    @Override
    public void suspend() {
        for (Synapse s : inputSynapses.values()) {
            s.getInput().removeActiveOutputSynapse(s);
        }
        for (Synapse s : outputSynapses.values()) {
            s.getOutput().removeActiveInputSynapse(s);
        }

        provider.lock.acquireReadLock();
        for (Synapse s : provider.activeInputSynapses.values()) {
            s.getInput().removeActiveOutputSynapse(s);
        }
        for (Synapse s : provider.activeOutputSynapses.values()) {
            s.getOutput().removeActiveInputSynapse(s);
        }
        provider.lock.releaseReadLock();
    }


    @Override
    public void reactivate() {
        provider.lock.acquireReadLock();
        for (Synapse s : provider.activeInputSynapses.values()) {
            s.getInput().addActiveOutputSynapse(s);
        }
        for (Synapse s : provider.activeOutputSynapses.values()) {
            s.getOutput().addActiveInputSynapse(s);
        }
        provider.lock.releaseReadLock();

        for (Synapse s : inputSynapses.values()) {
            s.getInput().addActiveOutputSynapse(s);
            if (!s.getInput().isSuspended()) {
                s.getOutput().addActiveInputSynapse(s);
            }
        }
        for (Synapse s : outputSynapses.values()) {
            s.getOutput().addActiveInputSynapse(s);
            if (!s.getOutput().isSuspended()) {
                s.getInput().addActiveOutputSynapse(s);
            }
        }
    }

    public void setBias(double b) {
        biasDelta = b - bias;
    }


    public void updateBiasDelta(double biasDelta) {
        this.biasDelta += biasDelta;
    }


    public double getBias() {
        return bias;
    }


    private double getBias(Synapse.State state) {
        return state == CURRENT ? bias : bias + biasDelta;
    }


    public double getNewBias() {
        return bias + biasDelta;
    }

    public double getBiasDelta() {
        return biasDelta;
    }


    public void register(Activation act) {
        Document doc = act.getDocument();

        if (addActivation(act)) {
            doc.addActivatedNeuron(act.getINeuron());
        }

        for(Map.Entry<Integer, Position> me: act.getSlots().entrySet()) {
            me.getValue().addActivation(me.getKey(), act);
        }

        doc.addActivation(act);
    }


    public static INeuron readNeuron(DataInput in, Neuron p) throws IOException {
        INeuron n = new INeuron();
        n.provider = p;
        n.threads = new ThreadState[p.getModel().numberOfThreads];
        n.readFields(in, p.getModel());
        return n;
    }


    public boolean isPassiveInputNeuron() {
        return passiveInputFunction != null;
    }


    public void registerPassiveInputSynapse(Synapse s) {
        if(passiveInputSynapses == null) {
            passiveInputSynapses = new TreeMap<>(Synapse.INPUT_SYNAPSE_COMP);
        }
        passiveInputSynapses.put(s, s);
    }


    public String toString() {
        return label;
    }


    public String toStringWithSynapses() {
        SortedSet<Synapse> is = new TreeSet<>((s1, s2) -> {
            int r = Double.compare(s2.getWeight(), s1.getWeight());
            if (r != 0) return r;
            return Integer.compare(s1.getInput().getId(), s2.getInput().getId());
        });

        is.addAll(inputSynapses.values());

        StringBuilder sb = new StringBuilder();
        sb.append(toString());
        sb.append("<");
        sb.append("B:");
        sb.append(Utils.round(bias));
        for (Synapse s : is) {
            sb.append(", ");
            sb.append(Utils.round(s.getWeight()));
            sb.append(":");
            sb.append(s.getInput().toString());
        }
        sb.append(">");
        return sb.toString();
    }


    public static class SynapseSummary implements Writable {
        private volatile double posDirSum;
        private volatile double negDirSum;
        private volatile double negRecSum;
        private volatile double posRecSum;
        private volatile double posPassiveSum;

        private volatile double posDirSumDelta = 0.0;
        private volatile double negDirSumDelta = 0.0;
        private volatile double negRecSumDelta = 0.0;
        private volatile double posRecSumDelta = 0.0;
        private volatile double posPassiveSumDelta = 0.0;


        public double getPosDirSum() {
            return posDirSum;
        }

        public double getNegDirSum() {
            return negDirSum;
        }

        public double getNegRecSum() {
            return negRecSum;
        }

        public double getPosRecSum() {
            return posRecSum;
        }

        public double getPosPassiveSum() {
            return posPassiveSum;
        }

        public double getPosSum(Synapse.State state) {
            return getPosDirSum(state) + getPosRecSum(state);
        }


        private double getPosDirSum(Synapse.State state) {
            return state == CURRENT ? posDirSum : posDirSum + posDirSumDelta;
        }

        private double getPosRecSum(Synapse.State state) {
            return state == CURRENT ? posRecSum : posRecSum + posRecSumDelta;
        }

        private double getPosPassiveSum(Synapse.State state) {
            return state == CURRENT ? posPassiveSum : posPassiveSum + posPassiveSumDelta;
        }


        public void updateSynapse(Synapse s) {
            if (!s.isInactive()) {
                updateSynapse(CURRENT, s);
                updateSynapse(NEXT, s);
            }
        }

        private void updateSynapse(Synapse.State state, Synapse s) {
            double sign = (state == CURRENT ? -1.0 : 1.0);

            updateSum(s.isRecurrent(), s.isNegative(state), sign * (s.getLimit(state) * s.getWeight(state)));

            if(s.getInput().get().isPassiveInputNeuron() && !s.isNegative(state)) {
                posPassiveSumDelta += sign * (!s.isNegative(state) ? (s.getLimit(state) * s.getWeight(state)) : 0.0);
            }
        }

        private void updateSum(boolean rec, boolean neg, double delta) {
            if(!rec) {
                if(!neg) {
                    posDirSumDelta += delta;
                } else {
                    negDirSumDelta += delta;
                }
            } else {
                if(!neg) {
                    posRecSumDelta += delta;
                } else {
                    negRecSumDelta += delta;
                }
            }
        }


        public void commit() {
            posDirSum += posDirSumDelta;
            negDirSum += negDirSumDelta;
            posRecSum += posRecSumDelta;
            negRecSum += negRecSumDelta;
            posPassiveSum += posPassiveSumDelta;

            posDirSumDelta = 0.0;
            negDirSumDelta = 0.0;
            negRecSumDelta = 0.0;
            posDirSumDelta = 0.0;
            posPassiveSumDelta = 0.0;
        }


        public static SynapseSummary read(DataInput in, Model m) throws IOException {
            SynapseSummary ss = new SynapseSummary();
            ss.readFields(in, m);
            return ss;
        }

        @Override
        public void write(DataOutput out) throws IOException {
            out.writeDouble(posDirSum);
            out.writeDouble(negDirSum);
            out.writeDouble(negRecSum);
            out.writeDouble(posRecSum);
            out.writeDouble(posPassiveSum);
        }

        @Override
        public void readFields(DataInput in, Model m) throws IOException {
            posDirSum = in.readDouble();
            negDirSum = in.readDouble();
            negRecSum = in.readDouble();
            posRecSum = in.readDouble();
            posPassiveSum = in.readDouble();
        }
    }
}
