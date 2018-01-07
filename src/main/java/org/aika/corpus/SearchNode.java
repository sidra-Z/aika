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
package org.aika.corpus;


import org.aika.Utils;
import org.aika.lattice.NodeActivation;
import org.aika.neuron.Activation.Rounds;
import org.aika.neuron.Activation.SynapseActivation;
import org.aika.neuron.Activation;
import org.aika.corpus.Conflicts.Conflict;
import org.aika.neuron.INeuron.NormWeight;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * The {@code SearchNode} class represents a node in the binary search tree that is used to find the optimal
 * interpretation for a given document. Each search node possess a refinement (simply a set of interpretation nodes).
 * The two options that this search node examines are that the refinement will either part of the final interpretation or not.
 * During each search step the activation values in all the neuron activations adjusted such that they reflect the interpretation of the current search path.
 * When the search reaches the maximum depth of the search tree and no further refinements exists, a weight is computed evaluating the current search path.
 * The search path with the highest weight is used to determine the final interpretation.
 *
 * <p> Before the search is started a set of initial refinements is generated from the conflicts within the document.
 * In other words, if there are no conflicts in a given document, then no search is needed. In this case the final interpretation
 * will simply be the set of all interpretation nodes. The initial refinements are then expanded, meaning all interpretation nodes that are consistent
 * with this refinement are added to the refinement. The initial refinements are then propagated along the search path as refinement candidates.
 *
 * @author Lukas Molzberger
 */
public class SearchNode implements Comparable<SearchNode> {

    private static final Logger log = LoggerFactory.getLogger(SearchNode.class);

    public static int MAX_SEARCH_STEPS = 100000;

    public int id;

    public SearchNode excludedParent;
    public SearchNode selectedParent;

    public long visited;
    List<InterprNode> refinement;
    Candidate candidate;
    int level;

    DebugState debugState;

    public enum DebugState {
        CACHED,
        LIMITED,
        EXPLORE
    }

    NormWeight weightDelta = NormWeight.ZERO_WEIGHT;
    NormWeight accumulatedWeight;

    public List<StateChange> modifiedActs = new ArrayList<>();


    public enum Coverage {
        SELECTED,
        UNKNOWN,
        EXCLUDED
    }


    public SearchNode(Document doc, SearchNode selParent, SearchNode exclParent, Candidate c, int level, List<InterprNode> changed) {
        id = doc.searchNodeIdCounter++;
        this.level = level;
        visited = doc.visitedCounter++;
        selectedParent = selParent;
        excludedParent = exclParent;

        if(c != null) {
            refinement = expandRefinement(Collections.singletonList(c.refinement), doc.visitedCounter++);
            candidate = c;
        }
        weightDelta = doc.vQueue.adjustWeight(this, changed);

        if (getParent() != null) {
            accumulatedWeight = weightDelta.add(getParent().accumulatedWeight);
        }
        if (Document.OPTIMIZE_DEBUG_OUTPUT) {
            log.info("Search Step: " + id + "  Candidate Weight Delta: " + weightDelta);
            log.info(doc.neuronActivationsToString(true, true, false) + "\n");
        }
    }


    private void collectResults(Collection<InterprNode> results) {
        if(refinement != null) {
            results.addAll(refinement);
        }
        if(selectedParent != null) selectedParent.collectResults(results);
    }


    public void computeBestInterpretation(Document doc) {
        ArrayList<InterprNode> results = new ArrayList<>();
        results.add(doc.bottom);

        int[] searchSteps = new int[1];

        List<InterprNode> rootRefs = expandRootRefinement(doc);
        refinement = expandRefinement(rootRefs, doc.visitedCounter++);

        if(Document.OPTIMIZE_DEBUG_OUTPUT) {
            log.info("Root SearchNode:" + toString());
        }


        Candidate[] candidates = generateCandidates(doc);

        Candidate c = candidates.length > level + 1 ? candidates[level + 1] : null;

        List<InterprNode> changed = new ArrayList<>();
        changed.addAll(refinement);

        markSelected(changed, refinement);
        markExcluded(changed, refinement);

        SearchNode child = new SearchNode(doc, this, null, c, level + 1, changed);
        child.search(doc, searchSteps, candidates);

        markUnselected(refinement);

        if (doc.selectedSearchNode != null) {
            doc.selectedSearchNode.reconstructSelectedResult(doc);
            doc.selectedSearchNode.collectResults(results);
        }

        doc.bestInterpretation = results;

        if(doc.interrupted) {
            log.warn("The search for the best interpretation has been interrupted. Too many search steps!");
        }
    }


    private void reconstructSelectedResult(Document doc) {
        if(getParent() != null) getParent().reconstructSelectedResult(doc);

        changeState(StateChange.Mode.NEW);

        for(StateChange sc : modifiedActs) {
            Activation act = sc.act;
            if(act.isFinalActivation()) {
                doc.finallyActivatedNeurons.add(act.key.node.neuron.get(doc));
            }
        }
    }


    public void dumpDebugState() {
        SearchNode n = this;
        while(n != null && n.level >= 0) {
            System.out.println(
                    n.level + " " +
                            n.debugState +
                            " CS:" + n.candidate.cache.size() +
                            " LIMITED:" + n.candidate.debugCounts[DebugState.LIMITED.ordinal()] +
                            " CACHED:" + n.candidate.debugCounts[DebugState.CACHED.ordinal()] +
                            " EXPLORE:" + n.candidate.debugCounts[DebugState.EXPLORE.ordinal()] +
                            " " + n.candidate.refinement.act.key.range +
                            " " + n.candidate.refinement.act.key.node.neuron.get().label
            );

            n = n.getParent();
        }
    }


    private NormWeight search(Document doc, int[] searchSteps, Candidate[] candidates) {
        if(candidate == null) {
            return processResult(doc);
        }

        NormWeight selectedWeight = NormWeight.ZERO_WEIGHT;
        NormWeight excludedWeight = NormWeight.ZERO_WEIGHT;

        boolean alreadySelected = checkSelected(refinement);
        boolean alreadyExcluded = checkExcluded(refinement, doc.visitedCounter++);

        if(searchSteps[0] > MAX_SEARCH_STEPS) {
            doc.interrupted = true;

            dumpDebugState();
        }
        searchSteps[0]++;

        if(Document.OPTIMIZE_DEBUG_OUTPUT) {
            log.info("Search Step: " + id);
            log.info(toString());
        }

        if(Document.OPTIMIZE_DEBUG_OUTPUT) {
            log.info(doc.neuronActivationsToString(true, true, false) + "\n");
        }

        if(alreadyExcluded || alreadySelected) {
            debugState = DebugState.LIMITED;
        } else {
            debugState = DebugState.EXPLORE;
        }

        CachedEntry cd = !alreadyExcluded && !alreadySelected ? getCachedDecision() : null;

        candidate.debugCounts[debugState.ordinal()]++;

        if (!alreadyExcluded) {
            List<InterprNode> changed = new ArrayList<>();
            changed.add(candidate.refinement);

            markSelected(changed, refinement);
            markExcluded(changed, refinement);

            if (cd == null || (cd.dir && accumulatedWeight.add(cd.weight).getNormWeight() >= getSelectedAccumulatedWeight(doc))) {
                Candidate c = candidates.length > level + 1 ? candidates[level + 1] : null;
                SearchNode child = new SearchNode(doc, this, excludedParent, c, level + 1, changed);
                selectedWeight = child.search(doc, searchSteps, candidates);
                child.changeState(StateChange.Mode.OLD);
            }

            markUnselected(refinement);
        }
        if(doc.interrupted) {
            return NormWeight.ZERO_WEIGHT;
        }

        if(!alreadySelected) {
            candidate.refinement.markedExcludedRefinement = true;
            List<InterprNode> changed = Collections.singletonList(candidate.refinement);

            if (cd == null || (!cd.dir && accumulatedWeight.add(cd.weight).getNormWeight() >= getSelectedAccumulatedWeight(doc))) {
                Candidate c = candidates.length > level + 1 ? candidates[level + 1] : null;
                SearchNode child = new SearchNode(doc, selectedParent, this, c, level + 1, changed);
                excludedWeight = child.search(doc, searchSteps, candidates);
                child.changeState(StateChange.Mode.OLD);
            }

            candidate.refinement.markedExcludedRefinement = false;
        }

        boolean dir = selectedWeight.getNormWeight() >= excludedWeight.getNormWeight();
        if(cd == null && !alreadyExcluded && !alreadySelected) {
            candidate.cache.put(this, new CachedEntry(dir, dir ? selectedWeight.sub(accumulatedWeight) : excludedWeight.sub(accumulatedWeight)));
        }

        return dir ? selectedWeight : excludedWeight;
    }


    private NormWeight processResult(Document doc) {
        double accNW = accumulatedWeight.getNormWeight();

        if (accNW > getSelectedAccumulatedWeight(doc)) {
            doc.selectedSearchNode = this;
        }

        return accumulatedWeight;
    }


    private double getSelectedAccumulatedWeight(Document doc) {
        return doc.selectedSearchNode != null ? doc.selectedSearchNode.accumulatedWeight.getNormWeight() : -1.0;
    }


    public Candidate[] generateCandidates(Document doc) {
        TreeSet<Candidate> candidates = new TreeSet<>();
        int i = 0;
        for(InterprNode cn: collectConflicts(doc)) {
            candidates.add(new Candidate(cn, i++));
        }
        return candidates.toArray(new Candidate[candidates.size()]);
    }



    private boolean checkSelected(List<InterprNode> n) {
        for(InterprNode x: n) {
            if(!isCovered(x.markedSelected)) return false;
        }
        return true;
    }


    private boolean checkExcluded(List<InterprNode> n, long v) {
        for(InterprNode x: n) {
            if(checkExcluded(x, v)) return true;
        }
        return false;
    }


    private boolean checkExcluded(InterprNode ref, long v) {
        if(ref.visitedCheckExcluded == v) return false;
        ref.visitedCheckExcluded = v;

        if(isCovered(ref.markedExcluded)) return true;

        for(InterprNode pn: ref.parents) {
            if(checkExcluded(pn, v)) return true;
        }

        return false;
    }


    public static Set<InterprNode> collectConflicts(Document doc) {
        Set<InterprNode> results = new TreeSet<>();
        long v = doc.visitedCounter++;
        for(InterprNode n: doc.bottom.children) {
            if(!n.conflicts.primary.isEmpty()) {
                results.add(n);
            }
            for(Conflict c: n.conflicts.secondary.values()) {
                results.add(c.secondary);
            }
        }
        return results;
    }


    private static List<InterprNode> expandRootRefinement(Document doc) {
        ArrayList<InterprNode> tmp = new ArrayList<>();
        tmp.add(doc.bottom);
        for(InterprNode pn: doc.bottom.children) {
            if(pn.fixed == Boolean.TRUE || (pn.isPrimitive() && pn.conflicts.primary.isEmpty() && pn.conflicts.secondary.isEmpty())) {
                tmp.add(pn);
            }
        }
        return tmp;
    }


    private List<InterprNode> expandRefinement(List<InterprNode> ref, long v) {
        ArrayList<InterprNode> tmp = new ArrayList<>();
        for(InterprNode n: ref) {
            markExpandRefinement(n, v);
            tmp.add(n);
        }

        for(InterprNode n: ref) {
            expandRefinementRecursiveStep(tmp, n, v);
        }

        if(ref.size() == tmp.size()) return tmp;
        else return expandRefinement(tmp, v);
    }


    private void markExpandRefinement(InterprNode n, long v) {
        if(n.markedExpandRefinement == v) return;
        n.markedExpandRefinement = v;

        for(InterprNode pn: n.parents) {
            markExpandRefinement(pn, v);
        }
    }


    private boolean hasUncoveredConflicts(InterprNode n) {
        if(!n.conflicts.hasConflicts()) return false;

        ArrayList<InterprNode> conflicts = new ArrayList<>();
        Conflicts.collectDirectConflicting(conflicts, n);
        for(InterprNode cn: conflicts) {
            if(!isCovered(cn.markedExcluded)) return true;
        }
        return false;
    }


    private void expandRefinementRecursiveStep(Collection<InterprNode> results, InterprNode n, long v) {
        if(n.visitedExpandRefinementRecursiveStep == v) return;
        n.visitedExpandRefinementRecursiveStep = v;

        if (n.refByOrInterprNode != null) {
            for (InterprNode on : n.refByOrInterprNode) {
                if(on.markedExpandRefinement != v && !hasUncoveredConflicts(on) && !isCovered(on.markedSelected)) {
                    markExpandRefinement(on, v);
                    results.add(on);
                }
            }
        }
        for(InterprNode pn: n.parents) {
            if(!pn.isBottom()) {
                expandRefinementRecursiveStep(results, pn, v);
            }
        }

        if(n.isBottom()) return;

        // Expand options that are partially covered by this refinement and partially by an earlier expand node.
        for(InterprNode cn: n.children) {
            if(cn.visitedExpandRefinementRecursiveStep == v) break;

            // Check if all parents are either contained in this refinement or an earlier refinement.
            boolean covered = true;
            for(InterprNode cnp: cn.parents) {
                if(cnp.visitedExpandRefinementRecursiveStep != v && !isCovered(cnp.markedSelected)) {
                    covered = false;
                    break;
                }
            }

            if(covered) {
                expandRefinementRecursiveStep(results, cn, v);
            }
        }
    }


    public Coverage getCoverage(InterprNode n) {
        if(n.fixed != null) {
            return n.fixed ? Coverage.SELECTED : Coverage.EXCLUDED;
        }
        if(n.markedExcludedRefinement) return Coverage.EXCLUDED;
        if(isCovered(n.markedSelected)) return Coverage.SELECTED;
        if(isCovered(n.markedExcluded)) return Coverage.EXCLUDED;
        return Coverage.UNKNOWN;
    }


    public boolean isCovered(long g) {
        SearchNode n = this;
        do {
            if(g == n.visited) return true;
            else if(g > n.visited) return false;
            n = n.selectedParent;
        } while(n != null);
        return false;
    }


    public void markSelected(List<InterprNode> changed, List<InterprNode> n) {
        for(InterprNode x: n) {
            markSelected(changed, x);
        }
    }


    public void markSelected(List<InterprNode> changed, InterprNode n) {
        if(isCovered(n.markedSelected)) return;

        n.markedSelected = visited;

        if(n.refByOrInterprNode != null) {
            for (InterprNode ref : n.refByOrInterprNode) {
                if(ref.selectedOrInterprNodes == null) {
                    ref.selectedOrInterprNodes = new TreeSet<>();
                }
                ref.selectedOrInterprNodes.add(n);
            }
        }

        if(n.isBottom()) {
            return;
        }

        if(changed != null) changed.add(n);

        return;
    }


    private void markUnselected(List<InterprNode> n) {
        for(InterprNode x: n) {
            if(x.markedSelected == visited) {
                if (x.refByOrInterprNode != null) {
                    for (InterprNode ref : x.refByOrInterprNode) {
                        ref.selectedOrInterprNodes.remove(x);
                    }
                }
            }
        }
    }


    private void markExcluded(List<InterprNode> changed, List<InterprNode> n) {
        for(InterprNode x: n) {
            markExcluded(changed, x);
        }
    }


    private void markExcluded(List<InterprNode> changed, InterprNode n) {
        List<InterprNode> conflicting = new ArrayList<>();
        Conflicts.collectAllConflicting(conflicting, n, n.doc.visitedCounter++);
        for(InterprNode cn: conflicting) {
            markExcludedRecursiveStep(changed, cn);
        }
    }


    private void markExcludedRecursiveStep(List<InterprNode> changed, InterprNode n) {
        if(isCovered(n.markedExcluded)) return;
        n.markedExcluded = visited;

        for(InterprNode c: n.children) {
            markExcludedRecursiveStep(changed, c);
        }

        // If the or option has two input options and one of them is already excluded, then when the other one is excluded we also have to exclude the or option.
        if(n.linkedByLCS != null) {
            for(InterprNode c: n.linkedByLCS) {
                if(checkOrNodeExcluded(c)) {
                    markExcludedRecursiveStep(changed, c);
                }
            }
        }

        if(changed != null) changed.add(n);

        return;
    }


    private boolean checkOrNodeExcluded(InterprNode n) {
        for(InterprNode on: n.orInterprNodes) {
            if(!isCovered(on.markedExcluded)) {
                return false;
            }
        }
        return true;
    }


    public String pathToString(Document doc) {
        return (selectedParent != null ? selectedParent.pathToString(doc) : "") + " - " + toString(doc);
    }


    public String toString(Document doc) {
        TreeSet<InterprNode> tmp = new TreeSet<>();
        for(InterprNode n: refinement) {
            n.collectPrimitiveNodes(tmp, doc.interpretationIdCounter++);
        }
        StringBuilder sb = new StringBuilder();
        for(InterprNode n: tmp) {
            sb.append(n.primId);
            sb.append(", ");
        }

        return sb.toString();
    }


    public void changeState(StateChange.Mode m) {
        for(StateChange sc: modifiedActs) {
            sc.restoreState(m);
        }
    }


    @Override
    public int compareTo(SearchNode sn) {
        return Integer.compare(id, sn.id);
    }


    /**
     * The {@code StateChange} class is used to store the state change of an activation that occurs in each node of
     * the binary search tree. When a candidate refinement is selected during the search, then the activation values of
     * all affected activation objects are adjusted. The changes to the activation values are also propagated through
     * the network. The old state needs to be stored here in order for the search to be able to restore the old network
     * state before following the alternative search branch.
     *
     */
    public static class StateChange {
        public Activation act;

        public Rounds oldRounds;
        public Rounds newRounds;

        public enum Mode { OLD, NEW }

        public static void saveOldState(List<StateChange> changes, Activation act, long v) {
            StateChange sc = act.currentStateChange;
            if(sc == null || act.currentStateV != v) {
                sc = new StateChange();
                sc.oldRounds = act.rounds.copy();
                act.currentStateChange = sc;
                act.currentStateV = v;
                sc.act = act;
                if(changes != null) {
                    changes.add(sc);
                }
            }
        }

        public static void saveNewState(Activation act) {
            StateChange sc = act.currentStateChange;

            sc.newRounds = act.rounds;
        }

        public void restoreState(Mode m) {
            act.rounds = (m == Mode.OLD ? oldRounds : newRounds).copy();
        }
    }


    public SearchNode getParent() {
        return getDecision() ? selectedParent : excludedParent;
    }


    private boolean getDecision() {
        return excludedParent == null || selectedParent.id > excludedParent.id;
    }


    public CachedEntry getCachedDecision() {
        x: for(Map.Entry<SearchNode, CachedEntry> me: candidate.cache.entrySet()) {
            SearchNode n = this;
            SearchNode cn = me.getKey();
            do {
                if(n.getDecision() != cn.getDecision()) {
                    if(affectsUnknown(n.getParent())) {
                        continue x;
                    }
                }
                n = n.getParent();
                cn = cn.getParent();
            } while(n.selectedParent != null);

            debugState = DebugState.CACHED;
            return me.getValue();
        }

        return null;
    }


    // TODO:
    public boolean affectsUnknown(SearchNode p) {
        for(InterprNode n: p.refinement) {
            if(n.act != null) {
                for(SynapseActivation sa: n.act.neuronOutputs) {
                    if(sa.isRecurrent() && !sa.synapse.isNegative()) {
                        if(getCoverage(sa.output.key.interpretation) == Coverage.UNKNOWN) return true;
                    }
                }
            }
        }
        return false;
    }


    private static class CachedEntry {
        boolean dir;
        NormWeight weight;

        private CachedEntry(boolean dir, NormWeight weight) {
            this.dir = dir;
            this.weight = weight;
        }
    }


    private static class Candidate implements Comparable<Candidate> {
        public TreeMap<SearchNode, CachedEntry> cache = new TreeMap<>();
        public InterprNode refinement;

        int[] debugCounts = new int[3];


        int id;
        int sequence = 0;
        int minBegin;
        int maxEnd;
        Integer minRid;

        public Candidate(InterprNode refinement, int id) {
            this.refinement = refinement;
            if(refinement.act != null) {
                sequence = refinement.act.getSequence();
                minBegin = refinement.act.key.range.begin;
                maxEnd = refinement.act.key.range.end;
                minRid = refinement.act.key.rid;
            } else {
                for(NodeActivation act: refinement.getActivations()) {
                    sequence = Math.max(sequence, refinement.act.getSequence());
                    if(act.key.range != null) {
                        minBegin= Math.min(minBegin, act.key.range.begin);
                        maxEnd =  Math.max(maxEnd, act.key.range.end);
                    }
                    minRid = Utils.nullSafeMin(minRid, act.key.rid);
                }
            }

            this.id = id;
        }


        @Override
        public int compareTo(Candidate c) {
            int r = Integer.compare(maxEnd, c.maxEnd);
            if(r != 0) return r;
            r = Integer.compare(c.minBegin, minBegin);
            if(r != 0) return r;

            r = Integer.compare(sequence, c.sequence);
            if(r != 0) return r;

            r = Utils.compareInteger(minRid, c.minRid);
            if(r != 0) return r;
            return Integer.compare(id, c.id);
        }
    }
}
