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


import network.aika.Document;
import network.aika.Model;
import network.aika.Provider;
import network.aika.Writable;
import network.aika.neuron.relation.Relation;
import network.aika.neuron.activation.Activation;
import network.aika.lattice.InputNode.InputActivation;
import network.aika.lattice.AndNode.AndActivation;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.*;


/**
 * The {@code InputNode} and the {@code AndNode} classes together form a pattern lattice, containing all
 * possible substructures of any given conjunction. For example if we have the conjunction ABCD where A, B, C, D are
 * the inputs, then the pattern lattice will contain the nodes ABCD, ABC, ABD, ACD, BCD, AB, AC, AD, BC, BD, CD,
 * A, B, C, D. The pattern lattice is organized in layers, where each layer only contains conjunctions/patterns of the
 * same size. These layers are connected through refinements. For example the and-node
 * ABD on layer 3 is connected to the and-node ABCD on layer 4 via the refinement C.
 *
 * @author Lukas Molzberger
 */
class AndNode extends Node<AndNode, AndActivation> {


    List<Entry> parents;


    public AndNode() {
        parents = new ArrayList<>();
    }


    public AndNode(Model m, int level, List<Entry> parents) {
        super(m, level);
        this.parents = parents;
    }


    private void init() {
        for(Entry e: parents) {
            e.rv.child = provider;
            Node pn = e.rv.parent.get();

            pn.addAndChild(e.ref, e.rv);
            pn.setModified();
        }
    }


    @Override
    protected void propagate(AndActivation act) {
        if (andChildren != null) {
            for (Link fl : act.inputs) {
                if(fl == null) continue;

                NodeActivation<?> pAct = fl.input;

                for (Link sl : pAct.outputsToAndNode.values()) {
                    NodeActivation secondAct = sl.output;
                    if (act != secondAct) {
                        applyIntern(act, fl.refAct, fl.ref, fl.rv, secondAct, sl.refAct, sl.ref, sl.rv);
                    }
                }
            }
        }

        propagateToOutputNeuron(act);
    }


    @Override
    public void cleanup() {
        if(!isRemoved && !isRequired()) {
            remove();

            for(Entry e: parents) {
                e.rv.parent.get().cleanup();
            }
        }
    }


    @Override
    void processActivation(AndActivation act) {
        if(act.isComplete()) {
            super.processActivation(act);
        }
    }



    private void applyIntern(AndActivation act, InputActivation refAct, Refinement ref, RefValue rv, NodeActivation secondAct, InputActivation secondRefAct, Refinement secondRef, RefValue secondRv) {
        Document doc = act.getDocument();

        lock.acquireReadLock();
        for(Map.Entry<Refinement, RefValue> me: andChildren.subMap(
                new Refinement(RelationsMap.MIN, secondRef.input),
                new Refinement(RelationsMap.MAX, secondRef.input)).entrySet()) {
            Refinement nRef = me.getKey();
            RefValue nRv = me.getValue();
            if(nRef.contains(secondRef, rv)) {
                AndNode nlNode = nRv.child.get(doc);

                AndActivation nlAct = lookupAndActivation(act, nRef);

                if(nlAct == null) {
                    nlAct = new AndActivation(doc, nlNode);
                    nlAct.link(nRef, nRv, secondRefAct, act);
                }

                nlAct.getNode().addActivation(nlAct);

                for(Entry secondNE: nlNode.parents) {
                    if(secondNE.rv.parent.get(doc) == secondAct.getNode() && secondNE.ref.contains(ref, secondRv)) {
                        nlAct.link(secondNE.ref, secondNE.rv, refAct, secondAct);
                        break;
                    }
                }
            }
        }
        lock.releaseReadLock();
    }


    private AndActivation lookupAndActivation(NodeActivation<?> input, Refinement ref) {
        for (Link l : input.outputsToAndNode.values()) {
            if(l.ref.compareTo(ref) == 0) {
                return l.output;
            }
        }
        return null;
    }


    RefValue expand(int threadId, Document doc, Refinement firstRef) {
        if(!firstRef.isConvertible()) return null;

        RefValue firstRV = getAndChild(firstRef);
        if(firstRV != null) {
            return firstRV;
        }

        int firstRefOffset = level;
        Integer[] firstOffsets = new Integer[level];
        for(int i = 0; i < firstOffsets.length; i++) {
            firstOffsets[i] = i;
        }

        List<Entry> nextLevelParents = new ArrayList<>();

        for(Entry firstParent: parents) {
            Node parentNode = firstParent.rv.parent.get(doc);

            Refinement secondParentRef = new Refinement(getParentRelations(firstRef, firstParent), firstRef.input);
            RefValue secondParentRV = parentNode.expand(threadId, doc, secondParentRef);

            if(secondParentRV == null) {
                continue;
            }

            Refinement secondRef = new Refinement(getRelations(firstRef, firstParent, secondParentRV), firstParent.ref.input);
            RefValue secondRV = new RefValue(getOffsets(firstRefOffset, firstParent, secondParentRV), firstOffsets[firstParent.rv.refOffset], secondParentRV.child);

            nextLevelParents.add(new Entry(secondRef, secondRV));
        }

        firstRV = new RefValue(firstOffsets, firstRefOffset, provider);
        nextLevelParents.add(new Entry(firstRef, firstRV));

        return createAndNode(provider.getModel(), doc, nextLevelParents, level + 1) ? firstRV : null;
    }


    private RelationsMap getParentRelations(Refinement firstRef, Entry firstParent) {
        Relation[] secondParentRelations = new Relation[firstRef.relations.length() - 1];
        for(int i = 0; i < firstRef.relations.length(); i++) {
            Integer j = firstParent.rv.reverseOffsets[i];
            if(j != null) {
                secondParentRelations[j] = firstRef.relations.get(i);
            }
        }
        return new RelationsMap(secondParentRelations);
    }


    private static RelationsMap getRelations(Refinement firstRef, Entry firstParent, RefValue secondParentRV) {
        Relation[] secondRelations = new Relation[firstParent.ref.relations.length() + 1];
        for(int i = 0; i < firstParent.ref.relations.length(); i++) {
            int j = secondParentRV.offsets[i];
            secondRelations[j] = firstParent.ref.relations.get(i);
        }

        Relation rel = firstRef.relations.get(firstParent.rv.refOffset);
        if(rel != null) {
            secondRelations[secondParentRV.refOffset] = rel.invert();
        }
        return new RelationsMap(secondRelations);
    }


    private static Integer[] getOffsets(int firstRefOffset, Entry firstParent, RefValue secondParentRV) {
        Integer[] secondOffsets = new Integer[secondParentRV.offsets.length + 1];
        for(int i = 0; i < firstParent.rv.reverseOffsets.length; i++) {
            Integer j = firstParent.rv.reverseOffsets[i];
            if(j != null) {
                secondOffsets[secondParentRV.offsets[j]] = i;
            }
        }
        secondOffsets[secondParentRV.refOffset] = firstRefOffset;
        return secondOffsets;
    }


    static boolean createAndNode(Model m, Document doc, List<Entry> parents, int level) {
        if (parents != null) {
            // Locking needs to take place in a predefined order.
            TreeSet<Provider<? extends Node>> parentsForLocking = new TreeSet();
            for(Entry e: parents) {
                parentsForLocking.add(e.rv.parent);
            }

            for (Provider<? extends Node> pn : parentsForLocking) {
                pn.get().lock.acquireWriteLock();
            }
            try {
                AndNode nln = new AndNode(m, level, parents);

                nln.init();
                nln.postCreate(doc);
            } finally {
                for (Provider<? extends Node> pn : parentsForLocking) {
                    pn.get().lock.releaseWriteLock();
                }
            }
        }

        return true;
    }


    @Override
    public void changeNumberOfNeuronRefs(int threadId, long v, int d) {
        super.changeNumberOfNeuronRefs(threadId, v, d);

        parents.forEach(e -> e.rv.parent.get().changeNumberOfNeuronRefs(threadId, v, d));
    }


    @Override
    public void reprocessInputs(Document doc) {
        for(Entry e: parents) {
            Node<?, NodeActivation<?>> pn = e.rv.parent.get();
            for(NodeActivation act : pn.getActivations(doc)) {
                act.repropagateV = markedCreated;
                act.getNode().propagate(act);
            }
        }
    }


    @Override
    public void remove() {
        super.remove();

        for(Entry e: parents) {
            Node pn = e.rv.parent.get();
            pn.lock.acquireWriteLock();
            pn.removeAndChild(e.ref);
            pn.setModified();
            pn.lock.releaseWriteLock();
        }
    }


    public String logicToString() {
        StringBuilder sb = new StringBuilder();
        sb.append("AND(" + level + ")[");
        boolean first = true;
        for(Entry e: parents) {
            if(!first) {
                sb.append(",");
            }
            first = false;
            sb.append(e.ref);
        }
        sb.append("]");
        return sb.toString();
    }


    @Override
    public void write(DataOutput out) throws IOException {
        out.writeBoolean(false);
        out.writeChar('A');
        super.write(out);

        out.writeInt(parents.size());
        for(Entry e: parents) {
            e.ref.write(out);
            e.rv.write(out);
        }
    }


    @Override
    public void readFields(DataInput in, Model m) throws IOException {
        super.readFields(in, m);

        int s = in.readInt();
        for(int i = 0; i < s; i++) {
            Refinement ref = Refinement.read(in, m);
            RefValue rv = RefValue.read(in, m);
            parents.add(new Entry(ref, rv));
        }
    }


    /**
     *
     */
    public static class Refinement implements Comparable<Refinement>, Writable {

        public RelationsMap relations;
        public Provider<InputNode> input;

        private Refinement() {}


        public Refinement(RelationsMap relations, Provider<InputNode> input) {
            this.relations = relations;
            this.input = input;
        }

        public boolean isConvertible() {
            for(Relation rel: relations.relations) {
                if(rel != null && rel.isConvertible()) return true;
            }
            return false;
        }

        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("(");
            sb.append(relations);
            sb.append(input.get().logicToString());
            sb.append(")");
            return sb.toString();
        }


        public void write(DataOutput out) throws IOException {
            relations.write(out);
            out.writeInt(input.getId());
        }


        public void readFields(DataInput in, Model m) throws IOException {
            relations = RelationsMap.read(in, m);
            input = m.lookupNodeProvider(in.readInt());
        }


        public static Refinement read(DataInput in, Model m) throws IOException {
            Refinement k = new Refinement();
            k.readFields(in, m);
            return k;
        }


        @Override
        public int compareTo(Refinement ref) {
            int r = input.compareTo(ref.input);
            if(r != 0) return r;

            return relations.compareTo(ref.relations);
        }

        public boolean contains(Refinement ref, RefValue rv) {
            for(int i = 0; i < ref.relations.length(); i++) {
                Relation ra = ref.relations.get(i);
                Relation rb = relations.get(rv.offsets[i]);

                if((ra == null && rb != null) || (ra != null && rb == null)) return false;

                if(ra != null && rb != null && ra.compareTo(rb) != 0) {
                    return false;
                }
            }

            return true;
        }
    }


    public static class RelationsMap implements Comparable<RelationsMap>, Writable {

        public static final RelationsMap MIN = new RelationsMap();
        public static final RelationsMap MAX = new RelationsMap();

        public Relation[] relations;


        public RelationsMap() {}


        public RelationsMap(Relation[] relations) {
            this.relations = relations;
        }


        public void write(DataOutput out) throws IOException {
            out.writeInt(relations.length);
            for(int i = 0; i < relations.length; i++) {
                Relation rel = relations[i];
                out.writeBoolean(rel != null);
                if(rel != null) {
                    rel.write(out);
                }
            }
        }


        public void readFields(DataInput in, Model m) throws IOException {
            int l = in.readInt();
            relations = new Relation[l];
            for(int i = 0; i < l; i++) {
                if(in.readBoolean()) {
                    relations[i] = Relation.read(in, m);
                }
            }
        }


        public static RelationsMap read(DataInput in, Model m) throws IOException {
            RelationsMap k = new RelationsMap();
            k.readFields(in, m);
            return k;
        }


        public String toString() {
            StringBuilder sb = new StringBuilder();
            for(int i = 0; i < relations.length; i++) {
                Relation rel = relations[i];
                if(rel != null) {
                    sb.append(i + ":" + rel + ", ");
                }
            }
            return sb.toString();
        }


        @Override
        public int compareTo(RelationsMap rm) {
            if (this == MIN) return -1;
            if (rm == MIN) return 1;
            if (this == MAX) return 1;
            if (rm == MAX) return -1;

            int r = Integer.compare(relations.length, rm.relations.length);
            if(r != 0) return r;

            for(int i = 0; i < relations.length; i++) {
                Relation ra = relations[i];
                Relation rb = rm.relations[i];

                if(ra == null && rb == null) continue;
                if(ra == null && rb != null) return -1;
                if(ra != null && rb == null) return 1;

                r = ra.compareTo(rb);
                if(r != 0) return r;
            }
            return 0;
        }

        public int length() {
            return relations.length;
        }

        public Relation get(int i) {
            return relations[i];
        }

        public int size() {
            if(relations.length == 0) return 0;
            int count = 0;
            for(int i = 0; i < relations.length; i++) {
                if(relations[i] != null) count++;
            }
            return count;
        }

        public boolean isExact() {
            for(Relation rel: relations) {
                if(!rel.isExact()) {
                    return false;
                }
            }
            return true;
        }
    }


    public static class RefValue implements Writable {
        public Integer[] offsets;  // input offsets -> output offsets
        public Integer[] reverseOffsets;  // output offsets -> input offsets
        public int refOffset;
        public Provider<? extends Node> parent;
        public Provider<AndNode> child;

        private RefValue() {}

        public RefValue(Integer[] offsets, int refOffset, Provider<? extends Node> parent) {
            this.offsets = offsets;
            reverseOffsets = new Integer[offsets.length + 1];
            for(int i = 0; i < offsets.length; i++) {
                reverseOffsets[offsets[i]] = i;
            }

            this.refOffset = refOffset;
            this.parent = parent;
        }

        @Override
        public void write(DataOutput out) throws IOException {
            out.writeInt(offsets.length);
            for(int i = 0; i < offsets.length; i++) {
                Integer ofs = offsets[i];
                out.writeBoolean(ofs != null);
                out.writeInt(ofs);
            }
            out.writeInt(refOffset);
            out.writeInt(parent.getId());
            out.writeInt(child.getId());
        }

        public static RefValue read(DataInput in, Model m)  throws IOException {
            RefValue rv = new RefValue();
            rv.readFields(in, m);
            return rv;
        }

        @Override
        public void readFields(DataInput in, Model m) throws IOException {
            int l = in.readInt();
            offsets = new Integer[l];
            reverseOffsets = new Integer[l + 1];
            for(int i = 0; i < l; i++) {
                if(in.readBoolean()) {
                    Integer ofs = in.readInt();
                    offsets[i] = ofs;
                    reverseOffsets[ofs] = i;
                }
            }
            refOffset = in.readInt();
            parent = m.lookupNodeProvider(in.readInt());
            child = m.lookupNodeProvider(in.readInt());
        }
    }


    public static class Entry {
        public Refinement ref;
        public RefValue rv;

        public Entry(Refinement ref, RefValue rv) {
            this.ref = ref;
            this.rv = rv;
        }
    }


    public static class AndActivation extends NodeActivation<AndNode> {

        public Link[] inputs;

        public AndActivation(Document doc, AndNode node) {
            super(doc, node);
            inputs = new Link[node.level];
        }

        public void link(Refinement ref, RefValue rv, InputActivation refAct, NodeActivation<?> input) {
            Link l = new Link(ref, rv, refAct, input, this);
            inputs[rv.refOffset] = l;
            input.outputsToAndNode.put(id, l);
        }

        public Activation getInputActivation(int i) {
            Link l = inputs[i];
            if(l != null) {
                return l.refAct.input;
            } else {
                for(int j = 0; j < inputs.length; j++) {
                    if (j != i) {
                        l = inputs[j];
                        if(l != null) {
                            return l.input.getInputActivation(l.rv.reverseOffsets[i]);
                        }
                    }
                }
                return null;
            }
        }

        public boolean isComplete() {
            int numberOfLinks = 0;
            for (Link l : inputs) {
                if (l != null) numberOfLinks++;
            }
            return getNode().parents.size() == numberOfLinks;
        }

        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("A-ACT(");
            boolean first = true;
            for(int i = 0; i < inputs.length; i++) {
                Activation iAct = getInputActivation(i);
                if(iAct != null) {
                    if(!first) {
                        sb.append(",");
                    }
                    sb.append(i + ":" + iAct.getLabel() + " " + iAct.slotsToString() + " (" + iAct.getId() + ")");

                    first = false;
                }
            }
            sb.append(")");
            return sb.toString();
        }
    }


    public static class Link {
        public Refinement ref;
        public RefValue rv;

        public NodeActivation<?> input;
        public InputActivation refAct;
        public AndActivation output;

        public Link(Refinement ref, RefValue rv, InputActivation refAct, NodeActivation<?> input, AndActivation output) {
            this.ref = ref;
            this.rv = rv;
            this.refAct = refAct;
            this.input = input;
            this.output = output;
        }
    }
}
