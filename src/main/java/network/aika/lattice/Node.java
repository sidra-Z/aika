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
package network.aika.lattice;


import network.aika.*;
import network.aika.Document;
import network.aika.neuron.INeuron;
import network.aika.neuron.Neuron;
import network.aika.neuron.Synapse;
import network.aika.neuron.activation.Activation;
import network.aika.neuron.activation.Linker;
import network.aika.neuron.activation.Position;
import network.aika.neuron.relation.Relation;

import java.io.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;


/**
 * The {@code Node} class is the abstract class for all the boolean logic nodes underneath the neural network layer.
 * These nodes form a boolean representation for all the neurons of the neural network. Whenever changes occur to the
 * synapse weights in the neural layer, then the structure of the boolean representation needs to be adjusted. Several
 * neurons, however, might share common substructures in this boolean representation. The {@code InputNode} and
 * the {@code AndNode} classes together form a pattern lattice, containing all possible substructures of any
 * given conjunction. For example if we have the conjunction ABCD where A, B, C, D are the inputs then the
 * pattern lattice will contain the nodes ABCD, ABC, ABD, ACD, BCD, AB, AC, AD, BC, BD, CD, A, B, C, D. The class
 * {@code OrNode} is a disjunction of either input-nodes or and-nodes. The or-node is connected with one of
 * the neurons.
 * <p>
 * <p>Each logic node has a set of activations. The activations are stored in the thread local data structure
 * {@code ThreadState}.
 *
 * @author Lukas Molzberger
 */
public abstract class Node<T extends Node, A extends NodeActivation<T>> extends AbstractNode<Provider<T>> implements Comparable<Node> {

    public static final Node MIN_NODE = new InputNode();
    public static final Node MAX_NODE = new InputNode();


    TreeMap<AndNode.Refinement, AndNode.RefValue> andChildren;
    TreeSet<OutputEntry> outputNeurons;

    int level;

    private AtomicInteger numberOfNeuronRefs = new AtomicInteger(0);
    volatile boolean isRemoved;

    // Only the children maps are locked.
    protected ReadWriteLock lock = new ReadWriteLock();

    private ThreadState<A>[] threads;

    long markedCreated;

    /**
     * Propagate an activation to the next node or the next neuron that is depending on the current node.
     *
     * @param act
     */
    protected abstract void propagate(A act);


    /**
     * The {@code ThreadState} is a thread local data structure containing the activations of a single document for
     * a specific logic node.
     */
    private static class ThreadState<A extends NodeActivation> {
        public long lastUsed;

        public List<A> added;
        public List<A> activations;

        public long visited;

        public boolean isQueued = false;
        public long queueId;

        public ThreadState() {
            added = new ArrayList<>();
            activations = new ArrayList<>();
        }
    }


    private ThreadState<A> getThreadState(int threadId, boolean create) {
        ThreadState<A> th = threads[threadId];
        if (th == null) {
            if (!create) return null;

            th = new ThreadState();
            threads[threadId] = th;
        }
        th.lastUsed = provider.getModel().docIdCounter.get();
        return th;
    }


    public void clearThreadState(int threadId, int deleteDocId) {
        Node.ThreadState th = threads[threadId];
        if (th != null && th.lastUsed < deleteDocId) {
            threads[threadId] = null;
        }
    }


    abstract AndNode.RefValue expand(int threadId, Document doc, AndNode.Refinement ref);

    public abstract void reprocessInputs(Document doc);

    public abstract void cleanup();

    public abstract String logicToString();


    protected Node() {
    }


    public Node(Model m, int level) {
        threads = new ThreadState[m.numberOfThreads];
        provider = new Provider(m, this);
        this.level = level;
        setModified();
    }


    public void postCreate(Document doc) {
        if(doc != null) {
            markedCreated = doc.createV;
            doc.addedNodes.add(this);
        }
    }


    void addOutputNeuron(int[] synapseIds, Neuron outputNeuron, int threadId) {
        changeNumberOfNeuronRefs(threadId, provider.getModel().visitedCounter.addAndGet(1), 1);

        OutputEntry oe = new OutputEntry(synapseIds, getProvider(), outputNeuron);
        addOutputEntry(oe);
        setModified();

        lock.acquireWriteLock();
        setModified();
        outputNeuron.get().setInputNode(oe);
        lock.releaseWriteLock();
    }



    void addOutputEntry(OutputEntry rv) {
        lock.acquireWriteLock();
        if (outputNeurons == null) {
            outputNeurons = new TreeSet<>();
        }
        outputNeurons.add(rv);
        lock.releaseWriteLock();
    }


    void removeOutputEntry(OutputEntry rv) {
        lock.acquireWriteLock();
        if (outputNeurons != null) {
            outputNeurons.remove(rv);
            if (outputNeurons.isEmpty()) {
                outputNeurons = null;
            }
        }
        lock.releaseWriteLock();
    }


    void addAndChild(AndNode.Refinement ref, AndNode.RefValue child) {
        if (andChildren == null) {
            andChildren = new TreeMap<>();
        }

        if(!andChildren.containsKey(ref)) {
            andChildren.put(ref, child);
        }
    }


    void removeAndChild(AndNode.Refinement ref) {
        if (andChildren != null) {
            andChildren.remove(ref);

            if (andChildren.isEmpty()) {
                andChildren = null;
            }
        }
    }


    void processActivation(A act) {
        register(act);
        propagate(act);
    }


    public void register(A act) {
        if(act.registered) {
            return;
        }

        Document doc = act.getDocument();

        assert act.getNode() == this;

        ThreadState th = getThreadState(doc.getThreadId(), true);
        if (th.activations.isEmpty()) {
            doc.addActivatedNode(act.getNode());
        }
        th.activations.add(act);

        doc.addedNodeActivations.add(act);
        act.registered = true;
    }


    public void clearActivations(Document doc) {
        clearActivations(doc.getThreadId());
    }


    public void clearActivations(int threadId) {
        ThreadState th = getThreadState(threadId, false);
        if (th == null) return;
        th.activations.clear();

        th.added.clear();
    }


    public void clearActivations() {
        for (int i = 0; i < provider.getModel().numberOfThreads; i++) {
            clearActivations(i);
        }
    }


    /**
     * Process all added or removed activation for this logic node.
     *
     * @param doc
     */
    public void processChanges(Document doc) {
        ThreadState th = getThreadState(doc.getThreadId(), true);
        List<A> tmpAdded = th.added;

        th.added = new ArrayList<>();

        tmpAdded.forEach(act -> processActivation(act));
    }


    protected void propagateToOutputNeuron(NodeActivation inputAct) {
        try {
            lock.acquireReadLock();
            if (outputNeurons != null) {
                for (OutputEntry oe : outputNeurons) {
                    propagate(oe, inputAct);
                }
            }
        } finally {
            lock.releaseReadLock();
        }
    }


    private void propagate(OutputEntry oe, NodeActivation inputAct) {
        Document doc = inputAct.getDocument();
        INeuron n = oe.child.get(doc);

        Activation act = new Activation(doc, n, oe.getSlots(inputAct));

        Link ol = new Link(oe, inputAct);
        act.setInputNodeLink(ol);
        ol.setOutput(act);

        doc.getUpperBoundQueue().add(act);

        ol.linkOutputActivation(act);
    }



    /**
     * Add a new activation to this logic node and further propagate this activation through the network.
     * This activation, however, will not be added immediately. This method only adds a request to the activations
     * queue in the document. The activation will be added when the method {@code Node.processChanges(Document doc)}
     * is called.
     *
     * @param act
     */
    public void addActivation(A act) {
        ThreadState<A> th = getThreadState(act.getThreadId(), true);
        th.added.add(act);
        act.getDocument().getNodeQueue().add(this);
    }


    public void remove() {
        assert !isRemoved;

        lock.acquireWriteLock();
        setModified();
        while (andChildren != null && !andChildren.isEmpty()) {
            andChildren.firstEntry().getValue().child.get().remove();
        }

        lock.releaseWriteLock();

        isRemoved = true;
    }


    AndNode.RefValue getAndChild(AndNode.Refinement ref) {
        lock.acquireReadLock();
        AndNode.RefValue result = andChildren != null ? andChildren.get(ref) : null;
        lock.releaseReadLock();
        return result;
    }


    public boolean isRequired() {
        return numberOfNeuronRefs.get() > 0;
    }


    protected void changeNumberOfNeuronRefs(int threadId, long v, int d) {
        ThreadState th = getThreadState(threadId, true);
        if (th.visited == v) return;
        th.visited = v;
        numberOfNeuronRefs.addAndGet(d);
    }


    public Collection<A> getActivations(Document doc) {
        ThreadState<A> th = getThreadState(doc.getThreadId(), false);
        if (th == null) return Collections.EMPTY_LIST;
        return th.activations;
    }



    public String getNeuronLabel() {
        return "";
    }


    public boolean isQueued(int threadId, long queueId) {
        ThreadState th = getThreadState(threadId, true);
        if (!th.isQueued) {
            th.isQueued = true;
            th.queueId = queueId;
        }
        return false;
    }


    public void setNotQueued(int threadId) {
        ThreadState th = getThreadState(threadId, false);
        if(th == null) return;
        th.isQueued = false;
    }


    public static int compareRank(int threadId, Node n1, Node n2) {
        int r = Integer.compare(n1.level, n2.level);
        if(r != 0) return r;

        ThreadState th1 = n1.getThreadState(threadId, true);
        ThreadState th2 = n2.getThreadState(threadId, true);
        return Long.compare(th1.queueId, th2.queueId);
    }


    public String toString() {
        if(this == MIN_NODE) return "MIN_NODE";
        if(this == MAX_NODE) return "MAX_NODE";

        StringBuilder sb = new StringBuilder();
        sb.append(getNeuronLabel());
        sb.append(" - ");
        sb.append(logicToString());
        return sb.toString();
    }


    public int compareTo(Node n) {
        if (this == n) return 0;
        if (this == MIN_NODE) return -1;
        if (n == MIN_NODE) return 1;
        if (this == MAX_NODE) return 1;
        if (n == MAX_NODE) return -1;

        return provider.compareTo(n.provider);
    }


    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(level);

        out.writeInt(numberOfNeuronRefs.get());

        if (andChildren != null) {
            out.writeInt(andChildren.size());
            for (Map.Entry<AndNode.Refinement, AndNode.RefValue> me : andChildren.entrySet()) {
                me.getKey().write(out);
                me.getValue().write(out);
            }
        } else {
            out.writeInt(0);
        }

        if (outputNeurons != null) {
            out.writeInt(outputNeurons.size());
            for (OutputEntry oe : outputNeurons) {
                oe.write(out);
            }
        } else {
            out.writeInt(0);
        }
    }


    @Override
    public void readFields(DataInput in, Model m) throws IOException {
        level = in.readInt();

        numberOfNeuronRefs.set(in.readInt());

        int s = in.readInt();
        for (int i = 0; i < s; i++) {
            addAndChild(AndNode.Refinement.read(in, m), AndNode.RefValue.read(in, m));
        }

        s = in.readInt();
        for (int i = 0; i < s; i++) {
            if (outputNeurons == null) {
                outputNeurons = new TreeSet<>();
            }
            outputNeurons.add(OutputEntry.read(in, m));
        }

        threads = new ThreadState[m.numberOfThreads];
    }


    public static Node readNode(DataInput in, Provider p) throws IOException {
        char type = in.readChar();
        Node n = null;
        switch (type) {
            case 'I':
                n = new InputNode();
                break;
            case 'A':
                n = new AndNode();
                break;
        }
        n.provider = p;

        n.readFields(in, p.getModel());
        return n;
    }


    public static class OutputEntry implements Comparable<OutputEntry>, Writable {
        int[] synapseIds;
        TreeMap<Integer, Integer> revSynapseIds = new TreeMap<>();
        Provider<? extends Node> parent;
        Neuron child;

        private OutputEntry() {}

        public OutputEntry(int[] synapseIds, Provider<? extends Node> parent, Neuron child) {
            this.synapseIds = synapseIds;
            for(int ofs = 0; ofs < synapseIds.length; ofs++) {
                revSynapseIds.put(synapseIds[ofs], ofs);
            }

            this.parent = parent;
            this.child = child;
        }


        SortedMap<Integer, Position> getSlots(NodeActivation inputAct) {
            SortedMap<Integer, Position> slots = new TreeMap<>();
            for(int i = 0; i < synapseIds.length; i++) {
                int synapseId = synapseIds[i];

                Synapse s = child.getSynapseById(synapseId);
                for(Map.Entry<Integer, Relation> me: s.getRelations().entrySet()) {
                    Relation rel = me.getValue();
                    if(me.getKey() == Synapse.OUTPUT) {
                        Activation iAct = inputAct.getInputActivation(i);
                        rel.mapSlots(slots, iAct);
                    }
                }
            }
            return slots;
        }


        @Override
        public void write(DataOutput out) throws IOException {
            out.writeInt(synapseIds.length);
            for(int i = 0; i < synapseIds.length; i++) {
                Integer ofs = synapseIds[i];
                out.writeBoolean(ofs != null);
                out.writeInt(ofs);
            }
            out.writeInt(parent.getId());
            out.writeInt(child.getId());
        }

        public static OutputEntry read(DataInput in, Model m)  throws IOException {
            OutputEntry rv = new OutputEntry();
            rv.readFields(in, m);
            return rv;
        }

        @Override
        public void readFields(DataInput in, Model m) throws IOException {
            int l = in.readInt();
            synapseIds = new int[l];
            for(int i = 0; i < l; i++) {
                if(in.readBoolean()) {
                    Integer ofs = in.readInt();
                    synapseIds[i] = ofs;
                    revSynapseIds.put(ofs, i);
                }
            }
            parent = m.lookupNodeProvider(in.readInt());
            child = m.lookupNeuron(in.readInt());
        }


        @Override
        public int compareTo(OutputEntry oe) {
            int r = child.compareTo(oe.child);
            if(r != 0) return r;

            r = parent.compareTo(oe.parent);
            if(r != 0) return r;

            r = Integer.compare(synapseIds.length, oe.synapseIds.length);
            if(r != 0) return r;

            for(int i = 0; i < synapseIds.length; i++) {
                r = Integer.compare(synapseIds[i], oe.synapseIds[i]);
                if(r != 0) return r;
            }
            return 0;
        }
    }


    public static class Link {
        public OutputEntry oe;

        private NodeActivation<?> input;
        private Activation output;

        public Link(OutputEntry oe, NodeActivation<?> input) {
            this.oe = oe;
            this.input = input;
        }


        public int size() {
            return oe.synapseIds.length;
        }

        public int get(int i) {
            return oe.synapseIds[i];
        }


        void linkOutputActivation(Activation act) {
            Linker l = act.getDocument().getLinker();
            for (int i = 0; i < size(); i++) {
                int synId = get(i);
                Synapse s = act.getSynapseById(synId);
                Activation iAct = input.getInputActivation(i);
                l.link(s, iAct, act);
            }
            l.process();
        }


        Collection<Activation.Link> getInputLinks(Neuron n) {
            List<Activation.Link> inputActs = new ArrayList<>();
            for (int i = 0; i < size(); i++) {
                int synId = get(i);
                Synapse s = n.getSynapseById(synId);
                Activation iAct = input.getInputActivation(i);
                inputActs.add(new Activation.Link(s, iAct, null));
            }
            return inputActs;
        }

        public void setOutput(Activation output) {
            this.output = output;
        }
    }

}
