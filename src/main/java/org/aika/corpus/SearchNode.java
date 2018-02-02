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
import org.aika.neuron.Activation.StateChange;
import org.aika.neuron.Activation.SynapseActivation;
import org.aika.neuron.Activation;
import org.aika.neuron.INeuron.NormWeight;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.aika.corpus.InterpretationNode.State.SELECTED;
import static org.aika.corpus.InterpretationNode.State.EXCLUDED;
import static org.aika.corpus.InterpretationNode.State.UNKNOWN;
import static org.aika.neuron.Activation.ACTIVATION_ID_COMP;

import java.util.*;

/**
 * The {@code SearchNode} class represents a node in the binary search tree that is used to find the optimal
 * interpretation for a given document. Each search node possess a refinement (simply a set of interpretation nodes).
 * The two options that this search node examines are that the refinement will either part of the final interpretation or not.
 * During each search step the activation values in all the neuron activations adjusted such that they reflect the interpretation of the current search path.
 * When the search reaches the maximum depth of the search tree and no further refinements exists, a weight is computed evaluating the current search path.
 * The search path with the highest weight is used to determine the final interpretation.
 * <p>
 * <p> Before the search is started a set of initial refinements is generated from the conflicts within the document.
 * In other words, if there are no conflicts in a given document, then no search is needed. In this case the final interpretation
 * will simply be the set of all interpretation nodes. The initial refinements are then expanded, meaning all interpretation nodes that are consistent
 * with this refinement are added to the refinement. The initial refinements are then propagated along the search path as refinement candidates.
 *
 * @author Lukas Molzberger
 */
public class SearchNode implements Comparable<SearchNode> {

    private static final Logger log = LoggerFactory.getLogger(SearchNode.class);

    public static int MAX_SEARCH_STEPS = Integer.MAX_VALUE;

    public int id;

    public SearchNode excludedParent;
    public SearchNode selectedParent;

    public long visited;
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

    public Map<Activation, StateChange> modifiedActs = new TreeMap<>(ACTIVATION_ID_COMP);



    Step step = Step.INIT;
    boolean alreadySelected;
    boolean alreadyExcluded;
    SearchNode selectedChild = null;
    SearchNode excludedChild = null;
    NormWeight selectedWeight = NormWeight.ZERO_WEIGHT;
    NormWeight excludedWeight = NormWeight.ZERO_WEIGHT;

    enum Step {
        INIT,
        PREPARE_SELECT,
        SELECT,
        POST_SELECT,
        PREPARE_EXCLUDE,
        EXCLUDE,
        POST_EXCLUDE,
        FINAL
    }


    public SearchNode(Document doc, SearchNode selParent, SearchNode exclParent, Candidate c, int level, Collection<InterpretationNode> changed) {
        id = doc.searchNodeIdCounter++;
        this.level = level;
        visited = doc.visitedCounter++;
        selectedParent = selParent;
        excludedParent = exclParent;

        SearchNode pn = getParent();
        SearchNode csn = null;
        boolean modified = true;
        if (c != null) {
            candidate = c;
            candidate.currentSearchNode = this;

            csn = candidate.cachedSearchNodes;

            if (csn == null || csn.getDecision() != getDecision()) {
                if (pn != null && pn.candidate != null) {
                    Activation act = pn.candidate.refinement.activation;
                    act.markDirty(visited);
                    for (SynapseActivation sa : act.neuronOutputs) {
                        sa.output.markDirty(visited);
                    }
                }
            } else {
                modified = csn.isModified();

                if (pn != null && pn.candidate != null && modified) {
                    pn.candidate.debugComputed[2]++;
                }
            }
            /*
            if(cached && !modified) {
                candidate.cachedSearchNode.changeState(StateChange.Mode.NEW);
                weightDelta = candidate.cachedSearchNode.weightDelta;

                if (getParent() != null) {
                    accumulatedWeight = weightDelta.add(getParent().accumulatedWeight);
                }
                return;

            }

            candidate.cachedSearchNode = this;
            */
        }

        weightDelta = doc.vQueue.adjustWeight(this, changed);

        if(modified) {
            markDirty(doc);
        }

        if (!modified) {
            if (Math.abs(weightDelta.w - csn.weightDelta.w) > 0.00001) {
                System.out.println();
            }
            if (!compareNewState(csn)) {
                System.out.println();
            }
        }

        if(candidate != null && modified) {
            candidate.cachedSearchNodes = this;
        }

        if (pn != null && pn.candidate != null) {
            pn.candidate.debugComputed[modified ? 1 : 0]++;
        }

        if (getParent() != null) {
            accumulatedWeight = weightDelta.add(getParent().accumulatedWeight);
        }
        if (Document.OPTIMIZE_DEBUG_OUTPUT) {
            log.info("Search Step: " + id + "  Candidate Weight Delta: " + weightDelta);
            log.info(doc.neuronActivationsToString(true, true, false) + "\n");
        }
    }


    private boolean isModified() {
        for (StateChange sc : modifiedActs.values()) {
            if (sc.getActivation().markedDirty > visited || sc.newState != sc.getActivation().key.interpretation.state) {
                return true;
            }
            if(sc.newRounds.isActive()) {
                for (SynapseActivation sa : sc.getActivation().neuronOutputs) {
                    if (sa.output.key.interpretation.state != UNKNOWN &&
                            sa.output.markedDirty > visited) {
                        return true;
                    }
                }
            }
        }
        return false;
    }


    private void markDirty(Document doc) {
        if(candidate == null) return;

        SearchNode csn = candidate.cachedSearchNodes;

        Set<Activation> acts = new TreeSet<>(ACTIVATION_ID_COMP);
        acts.addAll(modifiedActs.keySet());
        if(csn != null) {
            acts.addAll(csn.modifiedActs.keySet());
        }

        acts.forEach(act -> {
            StateChange sca = modifiedActs.get(act);
            StateChange scb = csn != null ? csn.modifiedActs.get(act) : null;

            if (sca == null || scb == null || !sca.newRounds.compare(scb.newRounds)) {
                for (Activation.SynapseActivation sa : act.neuronOutputs) {
                    sa.output.markDirty(visited);
                }
            }
        });
    }


    public boolean compareNewState(SearchNode cachedNode) {
        if (modifiedActs == null && cachedNode.modifiedActs == null) return true;
        if (modifiedActs == null || cachedNode.modifiedActs == null) return false;

        if (modifiedActs.size() != cachedNode.modifiedActs.size()) {
            return false;
        }
        for (Map.Entry<Activation, StateChange> me: modifiedActs.entrySet()) {
            StateChange sca = me.getValue();
            StateChange scb = cachedNode.modifiedActs.get(me.getKey());

            if (!sca.newRounds.compare(scb.newRounds)) {
                return false;
            }
        }

        return true;
    }


    public void collectResults(Collection<InterpretationNode> results) {
        if (candidate != null) {
            results.add(candidate.refinement);
        }
        if (selectedParent != null) selectedParent.collectResults(results);
    }


    public void reconstructSelectedResult(Document doc) {
        LinkedList<SearchNode> tmp = new LinkedList<>();
        SearchNode snt = this;
        do {
            tmp.addFirst(snt);
            snt = snt.getParent();
        } while(snt != null);

        for(SearchNode sn: tmp) {
            sn.changeState(Activation.Mode.NEW);

            SearchNode pn = sn.getParent();
            if (pn != null && pn.candidate != null) {
                pn.candidate.refinement.setState(sn.getDecision() ? SELECTED : EXCLUDED, sn.visited);
            }

            for (StateChange sc : sn.modifiedActs.values()) {
                Activation act = sc.getActivation();
                if (act.isFinalActivation()) {
                    doc.finallyActivatedNeurons.add(act.getINeuron());
                }
            }
        }
    }


    public void dumpDebugState() {
        SearchNode n = this;
        while (n != null && n.level >= 0) {
            System.out.println(
                    n.level + " " +
                            n.debugState +
                            " DECISION:" + n.getDecision() +
                            " " + n.candidate != null ? n.candidate.toString() : "" +
                            " MOD-ACTS:" + n.modifiedActs.size()
            );

            n = n.getParent();
        }
    }


    /**
     * This algorithm is the recursive version of the interpretation search.
     * Its perhaps easier to read than the iterative version.
     */
    public NormWeight searchRecursive(Document doc) {
        if (candidate == null) {
            return processResult(doc);
        }

        initStep(doc);

        if (prepareSelectStep(doc)) {
            selectedWeight = selectedChild.searchRecursive(doc);

            postReturn(selectedChild);
        }

        if (prepareExcludeStep(doc)) {
            excludedWeight = excludedChild.searchRecursive(doc);

            postReturn(excludedChild);
        }

        return finalStep();
    }


    /**
     * Searches for the best interpretation for the given document.
     *
     * This implementation of the algorithm is iterative to prevent stack overflow errors from happening.
     * Depending on the document the search tree might be getting very deep.
     *
     * @param doc
     * @param root
     */
    public static void searchIterative(Document doc, SearchNode root) {
        SearchNode sn = root;
        NormWeight returnWeight = null;

        do {
            switch(sn.step) {
                case INIT:
                    if (sn.candidate == null) {
                        returnWeight = sn.processResult(doc);
                        sn.step = Step.FINAL;
                        sn = sn.getParent();
                    } else {
                        sn.initStep(doc);
                        sn.step = Step.PREPARE_SELECT;
                    }
                    break;
                case PREPARE_SELECT:
                    sn.step = sn.prepareSelectStep(doc) ? Step.SELECT : Step.PREPARE_EXCLUDE;
                    break;
                case SELECT:
                    sn.step = Step.POST_SELECT;
                    sn = sn.selectedChild;
                    break;
                case POST_SELECT:
                    sn.selectedWeight = returnWeight;

                    sn.postReturn(sn.selectedChild);
                    sn.step = Step.PREPARE_EXCLUDE;
                    break;
                case PREPARE_EXCLUDE:
                    sn.step = sn.prepareExcludeStep(doc) ? Step.EXCLUDE : Step.FINAL;
                    break;
                case EXCLUDE:
                    sn.step = Step.POST_EXCLUDE;
                    sn = sn.excludedChild;
                    break;
                case POST_EXCLUDE:
                    sn.excludedWeight = returnWeight;

                    sn.postReturn(sn.excludedChild);
                    sn.step = Step.FINAL;
                    break;
                case FINAL:
                    returnWeight = sn.finalStep();
                    sn = sn.getParent();
                    break;
                default:
            }
        } while(sn.level >= root.level);
    }


    private void initStep(Document doc) {
        boolean precondition = checkPrecondition();

        alreadySelected = precondition && !candidate.isConflicting();
        alreadyExcluded = !precondition || checkExcluded(candidate.refinement, doc.visitedCounter++);

        if (doc.searchStepCounter > MAX_SEARCH_STEPS) {
            dumpDebugState();
            throw new RuntimeException("Max search step exceeded.");
        }

        doc.searchStepCounter++;

        if (Document.OPTIMIZE_DEBUG_OUTPUT) {
            log.info("Search Step: " + id);
            log.info(toString());
        }

        if (Document.OPTIMIZE_DEBUG_OUTPUT) {
            log.info(doc.neuronActivationsToString(true, true, false) + "\n");
        }

        if (alreadyExcluded || alreadySelected) {
            debugState = DebugState.LIMITED;
        } else if (getCachedDecision() != null) {
            debugState = DebugState.CACHED;
        } else {
            debugState = DebugState.EXPLORE;
        }

        candidate.debugCounts[debugState.ordinal()]++;
    }

    private Boolean getCachedDecision() {
        return !alreadyExcluded ? candidate.cachedDecision : null;
    }


    private boolean prepareSelectStep(Document doc) {
        if(alreadyExcluded || !(getCachedDecision() == null || getCachedDecision())) return false;

        candidate.refinement.setState(SELECTED, visited);

        if (candidate.cachedDecision == null) {
            invalidateCachedDecisions(doc.visitedCounter++);
        }

        Candidate c = doc.candidates.size() > level + 1 ? doc.candidates.get(level + 1) : null;
        selectedChild = new SearchNode(doc, this, excludedParent, c, level + 1, Collections.singleton(candidate.refinement));

        candidate.debugDecisionCounts[0]++;

        return true;
    }


    private boolean prepareExcludeStep(Document doc) {
        if(alreadySelected || !(getCachedDecision() == null || !getCachedDecision())) return false;

        candidate.refinement.setState(EXCLUDED, visited);

        Candidate c = doc.candidates.size() > level + 1 ? doc.candidates.get(level + 1) : null;
        excludedChild = new SearchNode(doc, selectedParent, this, c, level + 1, Collections.singleton(candidate.refinement));

        candidate.debugDecisionCounts[1]++;

        return true;
    }


    private void postReturn(SearchNode child) {
        child.changeState(Activation.Mode.OLD);

        candidate.refinement.setState(UNKNOWN, visited);
        candidate.refinement.activation.rounds.reset();
    }


    private NormWeight finalStep() {
        NormWeight result;
        if (getCachedDecision() == null) {
            boolean dir = selectedWeight.getNormWeight() >= excludedWeight.getNormWeight();
            dir = alreadySelected || (!alreadyExcluded && dir);

            if (!alreadyExcluded) {
                candidate.cachedDecision = dir;
            }

            result = dir ? selectedWeight : excludedWeight;
        } else {
            result = getCachedDecision() ? selectedWeight : excludedWeight;
        }

        selectedChild = null;
        excludedChild = null;
        return result;
    }



    private boolean checkPrecondition() {
        Set soin = candidate.refinement.selectedOrInterpretationNodes;
        return soin != null && !soin.isEmpty();
    }


    private void invalidateCachedDecisions(long v) {
        for (Activation act : candidate.refinement.neuronActivations) {
            for (SynapseActivation sa : act.neuronOutputs) {
                if (!sa.synapse.isNegative()) {
                    Candidate posCand = sa.output.key.interpretation.candidate;
                    if (posCand != null) {
                        if (posCand.cachedDecision == Boolean.FALSE && candidate.id > posCand.id) {
                            posCand.cachedDecision = null;
                        }
                    }

                    ArrayList<InterpretationNode> conflicting = new ArrayList<>();
                    Conflicts.collectConflicting(conflicting, sa.output.key.interpretation, v);
                    for (InterpretationNode c : conflicting) {
                        Candidate negCand = c.candidate;
                        if (negCand != null) {
                            if (negCand.cachedDecision == Boolean.TRUE && candidate.id > negCand.id) {
                                negCand.cachedDecision = null;
                            }
                        }
                    }
                }
            }
        }
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


    private boolean checkExcluded(InterpretationNode ref, long v) {
        ArrayList<InterpretationNode> conflicts = new ArrayList<>();
        Conflicts.collectConflicting(conflicts, ref, v);
        for (InterpretationNode cn : conflicts) {
            if (cn.state == SELECTED) return true;
        }
        return false;
    }



    public String pathToString(Document doc) {
        return (selectedParent != null ? selectedParent.pathToString(doc) : "") + " - " + toString(doc);
    }


    public String toString(Document doc) {
        TreeSet<InterpretationNode> tmp = new TreeSet<>();
        candidate.refinement.collectPrimitiveNodes(tmp, doc.interpretationIdCounter++);
        StringBuilder sb = new StringBuilder();
        for (InterpretationNode n : tmp) {
            sb.append(n.primId);
            sb.append(" Decision:" + getDecision());
            sb.append(", ");
        }

        return sb.toString();
    }


    public void changeState(Activation.Mode m) {
        for (StateChange sc : modifiedActs.values()) {
            sc.restoreState(m);
        }
    }


    @Override
    public int compareTo(SearchNode sn) {
        return Integer.compare(id, sn.id);
    }


    public SearchNode getParent() {
        return getDecision() ? selectedParent : excludedParent;
    }


    public boolean getDecision() {
        return excludedParent == null || selectedParent.id > excludedParent.id;
    }
}
