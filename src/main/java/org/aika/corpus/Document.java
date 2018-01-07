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


import org.aika.*;
import org.aika.lattice.*;
import org.aika.lattice.Node.ThreadState;
import org.aika.neuron.Activation;
import org.aika.neuron.Activation.State;
import org.aika.neuron.INeuron;
import org.aika.neuron.INeuron.NormWeight;
import org.aika.neuron.Synapse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * The {@code Document} class represents a single document which may be either used for processing a text or as
 * training input. A document consists of the raw text, the interpretations and the activations.
 *
 * <p>When the document is not needed any more, the method {@code clearActivations} must be called, since Aika only
 * supports a single document per thread and model.
 *
 * @author Lukas Molzberger
 */
public class Document implements Comparable<Document> {
    private static final Logger log = LoggerFactory.getLogger(Document.class);

    public static boolean APPLY_DEBUG_OUTPUT = false;
    public static boolean OPTIMIZE_DEBUG_OUTPUT = false;
    public static int CLEANUP_INTERVAL = 500;
    public static int MAX_ROUND = 20;

    public final int id;
    private final String content;

    public long visitedCounter = 1;
    public int interpretationIdCounter = 1;
    public int activationIdCounter = 0;
    public int searchNodeIdCounter = 0;

    public InterprNode bottom = new InterprNode(this, -1, 0, 0);

    public SearchNode selectedSearchNode = null;
    public List<InterprNode> bestInterpretation = null;

    public Model model;
    public int threadId;
    public boolean interrupted;

    public Queue queue = new Queue();
    public ValueQueue vQueue = new ValueQueue();
    public UpperBoundQueue ubQueue = new UpperBoundQueue();
    public BackPropagationQueue bQueue = new BackPropagationQueue();

    public TreeSet<INeuron> activatedNeurons = new TreeSet<>();
    public TreeSet<INeuron> finallyActivatedNeurons = new TreeSet<>();
    public TreeSet<Activation> inputNeuronActivations = new TreeSet<>();
    public TreeSet<Activation> targetActivations = new TreeSet<>();
    public TreeSet<Activation> errorSignalActivations = new TreeSet<>();
    public TreeMap<INeuron, Set<Synapse>> modifiedWeights = new TreeMap<>();

    public TreeMap<NodeActivation.Key, Activation> activationsByRid = new TreeMap<>((act1, act2) -> {
        int r = Integer.compare(act1.rid, act2.rid);
        if (r != 0) return r;
        return act1.compareTo(act2);
    });
    public TreeSet<Node> addedNodes = new TreeSet<>();


    public static Comparator<NodeActivation> ACTIVATIONS_OUTPUT_COMPARATOR = (act1, act2) -> {
        int r = Range.compare(act1.key.range, act2.key.range, false);
        if (r != 0) return r;
        r = Utils.compareInteger(act1.key.rid, act2.key.rid);
        if (r != 0) return r;
        r = act1.key.interpretation.compareTo(act2.key.interpretation);
        if (r != 0) return r;
        return act1.key.node.compareTo(act2.key.node);
    };


    public Document(int id, String content, Model model, int threadId) {
        this.id = id;
        this.content = content;

        this.model = model;
        this.threadId = threadId;
    }


    public String getContent() {
        return content;
    }


    public int length() {
        return content.length();
    }


    public String toString() {
		return content;
	}


    public String getText(Range r) {
        return content.substring(
                Math.max(0, Math.min(r.begin, length())),
                Math.max(0, Math.min(r.end, length()))
        );
    }


    public String bestInterpretationToString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Best Interpretation:\n");
        sb.append(bestInterpretation.toString());
        sb.append("\n");
        return sb.toString();
    }


    @Override
    public int compareTo(Document doc) {
        return Integer.compare(id, doc.id);
    }


    public void propagate() {
        boolean flag = true;
        while(flag) {
            queue.processChanges();
            flag = ubQueue.process();
        }
    }


    /**
     * The method <code>process</code> needs to be called after all the input activations have been added to the
     * network. It performs the search for the best interpretation.
     */
    public void process() {
        inputNeuronActivations.forEach(act -> vQueue.propagateWeight(0, act));
        interrupted = false;
        SearchNode root = new SearchNode(this, null, null, null, -1, Collections.emptyList());
        root.computeBestInterpretation(this);
    }


    public void notifyWeightsModified(INeuron n, Collection<Synapse> inputSynapses) {
        Set<Synapse> is = modifiedWeights.get(n);
        if(is == null) {
            is = new TreeSet<>(Synapse.INPUT_SYNAPSE_COMP);
            modifiedWeights.put(n, is);
        }
        is.addAll(inputSynapses);
    }


    /**
     * Updates the model after the training step.
     * It applies the weight and bias delta values and reflects the changes in the logic node structure.
     */
    public void commit() {
        modifiedWeights.forEach((n, inputSyns) -> Converter.convert(model, threadId, n, inputSyns));
    }


    public static class DiscoveryConfig {
        public PatternEvaluation checkValidPattern;
        public ActivationEvaluation checkExpandable;
        public Counter counter;


        public DiscoveryConfig setCheckValidPattern(PatternEvaluation checkValidPattern) {
            this.checkValidPattern = checkValidPattern;
            return this;
        }


        /**
         * This callback checks whether the current pattern might be refined to an even larger pattern.
         * If frequency is the criterion, then infrequent are not expandable.
         *
         * @param checkExpandable
         * @return
         */
        public DiscoveryConfig setCheckExpandable(ActivationEvaluation checkExpandable) {
            this.checkExpandable = checkExpandable;
            return this;
        }


        /**
         * The counter callback function should implement a customized counting function.
         * The counting function should modify the custom meta object stored in the node.
         * The NodeStatisticFactory is used to instantiate the custom meta object for a node.
         *
         * @param counter
         * @return
         */
        public DiscoveryConfig setCounter(Counter counter) {
            this.counter = counter;
            return this;
        }
    }


    public void discoverPatterns(DiscoveryConfig discoveryConfig) {
        Collection<NodeActivation> allActs = getAllNodeActivations();
        allActs.forEach(act -> discoveryConfig.counter.count(act));

        allActs.stream()
                .filter(act -> discoveryConfig.checkExpandable.evaluate(act))
                .forEach(act -> act.key.node.discover(this, act, discoveryConfig));
    }


    public Collection<NodeActivation> getAllNodeActivations() {
        long v = visitedCounter++;
        TreeSet<NodeActivation> results = new TreeSet<>();
        for(INeuron n: activatedNeurons) {
            for(Activation act: n.getAllActivations(this)) {
                collectNodeActivations(results, act, v);
            }
        }
        return results;
    }


    private void collectNodeActivations(Collection<NodeActivation> results, NodeActivation<?> act, long v) {
        if(act.visited == v) return;
        act.visited = v;

        results.add(act);

        for(NodeActivation<?> oAct: act.outputs.values()) {
            collectNodeActivations(results, oAct, v);
        }
    }


    public static class TrainConfig {
        public SynapseEvaluation synapseEvaluation;
        public double learnRate;
        public boolean performBackpropagation;


        /**
         * Determines whether a synapse should be created between two neurons during training.
         *
         * @param synapseEvaluation
         * @return
         */
        public TrainConfig setSynapseEvaluation(SynapseEvaluation synapseEvaluation) {
            this.synapseEvaluation = synapseEvaluation;
            return this;
        }


        public TrainConfig setLearnRate(double learnRate) {
            this.learnRate = learnRate;
            return this;
        }


        public TrainConfig setPerformBackpropagation(boolean performBackpropagation) {
            this.performBackpropagation = performBackpropagation;
            return this;
        }
    }


    public void train(TrainConfig trainConfig) {
        targetActivations.forEach(tAct -> tAct.key.node.neuron.get(this).computeOutputErrorSignal(tAct));

        if(trainConfig.performBackpropagation) {
            bQueue.backpropagtion();
        }

        for (Activation act : errorSignalActivations) {
            act.key.node.neuron.get(this).train(this, act, trainConfig.learnRate, trainConfig.synapseEvaluation);
        }
        errorSignalActivations.clear();
    }

    /**
     * Removes the activations of this document from the model again.
     */
    public void clearActivations() {
        activatedNeurons.forEach(n -> n.clearActivations(this));

        activatedNeurons.clear();
        addedNodes.clear();

        if(model.lastCleanup[threadId] + CLEANUP_INTERVAL < id) {
            model.lastCleanup[threadId] = id;

            List<Provider<? extends AbstractNode>> tmp;
            synchronized(model.activeProviders) {
                tmp = new ArrayList<>(model.activeProviders.values());
            }

            tmp.forEach(np -> {
                AbstractNode an = np.getIfNotSuspended();
                if (an != null && an instanceof Node) {
                    Node n = (Node) an;
                    Node.ThreadState th = n.threads[threadId];
                    if (th != null && th.lastUsed + CLEANUP_INTERVAL < id) {
                        n.threads[threadId] = null;
                    }
                }
            });
        }

        model.docs[threadId] = null;
    }


    public String generateOutputText() {
        StringBuilder sb = new StringBuilder();
        finallyActivatedNeurons.stream()
                .filter(n -> n.outputText != null)
                .forEach(n -> {
            for (Activation act : n.getFinalActivations(this)) {
                sb.replace(act.key.range.begin, act.key.range.end, n.outputText);
            }
        });

        return sb.toString();
    }


    public String neuronActivationsToString(boolean withWeights) {
        return neuronActivationsToString(withWeights, false, false);
    }


    public String neuronActivationsToString(boolean withWeights, boolean withTextSnipped, boolean withLogic) {
        return neuronActivationsToString(null, withWeights, withTextSnipped, withLogic);
    }


    public String neuronActivationsToString(SearchNode sn, boolean withWeights, boolean withTextSnipped, boolean withLogic) {
        Set<Activation> acts = new TreeSet<>(ACTIVATIONS_OUTPUT_COMPARATOR);

        for (INeuron n : activatedNeurons) {
            Stream<Activation> s = Activation.select(this, n, null, null, null, null, InterprNode.Relation.CONTAINED_IN);
            acts.addAll(s.collect(Collectors.toList()));
        }

        StringBuilder sb = new StringBuilder();
        for(Activation act: acts) {
            if(act.upperBound <= 0.0 && (act.targetValue == null || act.targetValue <= 0.0)) {
                continue;
            }

            sb.append(act.id + " ");

            if(sn != null) {
                sb.append(sn.getCoverage(act.key.interpretation) + " ");
                sb.append(act.sequence + " ");
            }

            sb.append(act.key.range);

            if(withTextSnipped) {
                sb.append(" ");
                if(act.key.node.neuron.get().outputText != null) {
                    sb.append(collapseText(act.key.node.neuron.get().outputText));
                } else {
                    sb.append(collapseText(getText(act.key.range)));
                }
            }
            sb.append(" - ");

            sb.append(act.key.interpretation);
            sb.append(" - ");

            sb.append(withLogic ? act.key.node.toString() : act.key.node.getNeuronLabel());

            sb.append(" - Rid:");
            sb.append(act.key.rid);

            sb.append(" - UB:");
            sb.append(Utils.round(act.upperBound));
            if (withWeights) {
                sb.append(" - ");
                for(Map.Entry<Integer, Activation.State> me: act.rounds.rounds.entrySet()) {
                    Activation.State s = me.getValue();
                    sb.append("[R:" + me.getKey());
                    sb.append(" VALUE:" + Utils.round(s.value));
                    sb.append(" W:" + Utils.round(s.weight.w));
                    sb.append(" N:" + Utils.round(s.weight.n));
                    sb.append("]");
                }

                if (act.isFinalActivation()) {
                    State fs = act.getFinalState();
                    sb.append(" - FV:" + Utils.round(fs.value));
                    sb.append(" FW:" + Utils.round(fs.weight.w));
                    sb.append(" FN:" + Utils.round(fs.weight.n));

                    if(act.targetValue != null) {
                        sb.append(" - TV:" + Utils.round(act.targetValue));
                    }
                }
            }
            sb.append("\n");
        }

        if(selectedSearchNode != null) {
            sb.append("\n Final SearchNode:" + selectedSearchNode.id + "  WeightSum:" + selectedSearchNode.accumulatedWeight.toString() + "\n");
        }
        return sb.toString();
    }


    private String collapseText(String txt) {
        if (txt.length() <= 10) {
            return txt;
        } else {
            return txt.substring(0, 5) + "..." + txt.substring(txt.length() - 5);
        }
    }


    public class Queue {

        public final TreeSet<Node> queue = new TreeSet<>(new Comparator<Node>() {
            @Override
            public int compare(Node n1, Node n2) {
                int r = Integer.compare(n1.level, n2.level);
                if(r != 0) return r;

                ThreadState th1 = n1.getThreadState(threadId, true);
                ThreadState th2 = n2.getThreadState(threadId, true);
                return Long.compare(th1.queueId, th2.queueId);
            }
        });

        private long queueIdCounter = 0;


        public void add(Node n) {
            ThreadState th = n.getThreadState(threadId, true);

            if(!th.isQueued) {
                th.isQueued = true;
                th.queueId = queueIdCounter++;
                queue.add(n);
            }
        }


        public void processChanges() {
            while(!queue.isEmpty()) {
                Node n = queue.pollFirst();
                ThreadState th = n.getThreadState(threadId, true);

                th.isQueued = false;
                n.processChanges(Document.this);

                if(APPLY_DEBUG_OUTPUT) {
                    log.info("QueueId:" + th.queueId);
                    log.info(n.toString() + "\n");
                    log.info("\n" + neuronActivationsToString( true, true, true));
                }
            }
        }
    }


    public class UpperBoundQueue {
        public final ArrayDeque<Activation> queue = new ArrayDeque<>();


        public void add(Activation act) {
            if(!act.ubQueued) {
                act.ubQueued = true;
                queue.addLast(act);
            }
        }


        public boolean process() {
            boolean flag = false;
            while(!queue.isEmpty()) {
                flag = true;
                Activation act = queue.pollFirst();
                act.ubQueued = false;

                double oldUpperBound = act.upperBound;

                INeuron n = act.key.node.neuron.get(Document.this);

                if(act.inputValue == null) {
                    n.computeBounds(act);
                } else {
                    act.upperBound = act.inputValue;
                    act.lowerBound = act.inputValue;
                }

                if(Math.abs(act.upperBound - oldUpperBound) > 0.01) {
                    for(Activation.SynapseActivation sa: act.neuronOutputs) {
                        add(sa.output);
                    }
                }

                if(oldUpperBound <= 0.0 && act.upperBound > 0.0) {
                    for(Provider<InputNode> out: n.outputNodes.values()) {
                        out.get(Document.this).addActivation(Document.this, act);
                    }
                }
            }
            return flag;
        }

    }


    private static Comparator<Activation> VALUE_QUEUE_COMP = (a, b) -> {
        int r = Integer.compare(a.getSequence(), b.getSequence());
        if(r != 0) return r;
        return Integer.compare(a.id, b.id);
    };


    public class ValueQueue {


        public final ArrayList<TreeSet<Activation>> queue = new ArrayList<>();

        public void propagateWeight(int round, Activation act)  {
            for(Activation.SynapseActivation sa: act.neuronOutputs) {
                int r = sa.isRecurrent() ? round + 1 : round;
                add(r, sa.output);
            }
        }


        public NormWeight adjustWeight(SearchNode cand, List<InterprNode> changed) {
            long v = visitedCounter++;

            for(InterprNode n: changed) {
                addAllActs(n.getNeuronActivations());

                // Does not need to be expanded recursively, because the activation will be propagated anyway.
                if(n.refByOrInterprNode != null) {
                    for (InterprNode on: n.refByOrInterprNode) {
                        addAllActs(on.getNeuronActivations());
                    }
                }
            }

            return processChanges(cand, v);
        }


        private void addAllActs(Collection<Activation> acts) {
            for(Activation act: acts) {
                add(0, act);
                for(Activation.SynapseActivation sa: act.neuronOutputs) {
                    if(sa.isRecurrent()) {
                        add(0, sa.output);
                    }
                }
            }
        }


        public void add(int round, Activation act) {
            if(act.rounds.isQueued(round)) return;

            TreeSet<Activation> q;
            if(round < queue.size()) {
                q = queue.get(round);
            } else {
                assert round == queue.size();
                q = new TreeSet<>(VALUE_QUEUE_COMP);
                queue.add(q);
            }

            act.rounds.setQueued(round, true);
            q.add(act);
        }


        public INeuron.NormWeight processChanges(SearchNode sn, long v) {
            NormWeight delta = NormWeight.ZERO_WEIGHT;
            for(int round = 0; round < queue.size(); round++) {
                TreeSet<Activation> q = queue.get(round);
                while (!q.isEmpty()) {
                    Activation act = q.pollLast();
                    act.rounds.setQueued(round, false);

                    State s;
                    if(act.inputValue != null) {
                        s = new State(act.inputValue, 0, NormWeight.ZERO_WEIGHT);
                    } else {
                        s = act.key.node.neuron.get(Document.this).computeWeight(round, act, sn);
                    }

                    if (OPTIMIZE_DEBUG_OUTPUT) {
                        log.info(act.key + " Round:" + round);
                        log.info("Value:" + s.value + "  Weight:" + s.weight.w + "  Norm:" + s.weight.n + "\n");
                    }

                    if (round == 0 || !act.rounds.get(round).equalsWithWeights(s)) {
                        SearchNode.StateChange.saveOldState(sn.modifiedActs, act, v);

                        State oldState = act.rounds.get(round);

                        boolean propagate = act.rounds.set(round, s);

                        SearchNode.StateChange.saveNewState(act);

                        if (propagate) {
                            if(round > MAX_ROUND) {
                                log.error("Error: Maximum number of rounds reached. The network might be oscillating.");
                                log.info(neuronActivationsToString(sn, true, true, true));

                                dumpOscillatingActivations(sn);
                                throw new RuntimeException("Maximum number of rounds reached. The network might be oscillating.");
                            } else {
                                propagateWeight(round, act);
                            }
                        }

                        if (round == 0) {
                            // In case that there is a positive feedback loop.
                            add(1, act);
                        }

                        if (act.rounds.getLastRound() != null && round >= act.rounds.getLastRound()) { // Consider only the final round.
                            delta = delta.add(s.weight.sub(oldState.weight));
                        }
                    }
                }
            }
            return delta;
        }
    }


    private void dumpOscillatingActivations(SearchNode sn) {
        activatedNeurons.stream()
                .flatMap(n -> n.getAllActivations(this).stream())
                .filter(act -> act.rounds.getLastRound() != null && act.rounds.getLastRound() > MAX_ROUND - 5)
                .forEach(act -> log.error(act.key + " " + sn.getCoverage(act.key.interpretation) + " " + act.rounds));
    }


    public class BackPropagationQueue {

        public final TreeSet<Activation> queue = new TreeSet<>(new Comparator<Activation>() {
            @Override
            public int compare(Activation act1, Activation act2) {
                Activation.State fs1 = act1.getFinalState();
                Activation.State fs2 = act2.getFinalState();

                int r = Integer.compare(fs2.fired, fs1.fired);
                if(r != 0) return r;
                return act1.key.compareTo(act2.key);
            }
        });

        private long queueIdCounter = 0;


        public void add(Activation act) {
            if(!act.isQueued) {
                act.isQueued = true;
                act.queueId = queueIdCounter++;
                queue.add(act);
            }
        }


        public void backpropagtion() {
            while(!queue.isEmpty()) {
                Activation act = queue.pollFirst();

                act.isQueued = false;
                act.key.node.neuron.get(Document.this).computeBackpropagationErrorSignal(act);
            }
        }
    }


    public interface PatternEvaluation {

        /**
         * Check if <code>node</code> is an interesting pattern that might be considered for further processing.
         *
         * This property is required to be monotonic over the size of the pattern. In other words, if a pattern is
         * interesting, then all its sub patterns also need to be interesting.
         *
         * @param n
         * @return
         */

        boolean evaluate(Node n);
    }


    public interface ActivationEvaluation {

        /**
         * Check if <code>node</code> is an interesting pattern that might be considered for further processing.
         *
         * This property is required to be monotonic over the size of the pattern. In other words, if a pattern is
         * interesting, then all its sub patterns also need to be interesting.
         *
         * @param act
         * @return
         */

        boolean evaluate(NodeActivation act);
    }


    public interface SynapseEvaluation {

        /**
         * Determines whether a synapse should be created between two neurons during training.
         *
         * @param iAct
         * @param oAct
         * @return
         */
        SynEvalResult evaluate(Activation iAct, Activation oAct);
    }


    public static class SynEvalResult {
        public SynEvalResult(Synapse.Key synapseKey, double significance) {
            this.synapseKey = synapseKey;
            this.significance = significance;
        }

        public Synapse.Key synapseKey;
        public double significance;
    }


    public interface Counter {

        /**
         * Updates the statistics of this node
         *
         * @param act
         * @return
         */
        void count(NodeActivation act);
    }
}
