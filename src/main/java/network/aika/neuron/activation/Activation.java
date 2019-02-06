package network.aika.neuron.activation;

import network.aika.Document;
import network.aika.Model;
import network.aika.Utils;
import network.aika.Writable;
import network.aika.lattice.OrNode;
import network.aika.lattice.OrNode.OrActivation;
import network.aika.neuron.INeuron;
import network.aika.neuron.INeuron.Type;
import network.aika.neuron.Neuron;
import network.aika.neuron.Synapse;
import network.aika.neuron.activation.SearchNode.Decision;
import network.aika.neuron.relation.Relation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Stream;

import static network.aika.neuron.activation.Linker.Direction.INPUT;
import static network.aika.neuron.activation.Linker.Direction.OUTPUT;
import static network.aika.neuron.activation.SearchNode.Decision.EXCLUDED;
import static network.aika.neuron.activation.SearchNode.Decision.SELECTED;
import static network.aika.neuron.activation.Activation.Link.INPUT_COMP;
import static network.aika.neuron.activation.Activation.Link.OUTPUT_COMP;
import static network.aika.neuron.INeuron.ALLOW_WEAK_NEGATIVE_WEIGHTS;
import static network.aika.neuron.activation.SearchNode.Decision.UNKNOWN;


/**
 * The {@code Activation} class is the most central class in Aika. On the one hand it stores the activation value
 * for a given neuron in the {@code State} substructure. On the other hand it specifies where this activation is
 * located within the document and to which interpretation it belongs. The {@code Activation.Key} therefore
 * consists of the logic node to which this activation belongs. If this logic node is an or-node, then this activation
 * automatically also belongs to the neuron as well. Furthermore, the key contains the char range within the document
 * and the relational id (rid). The relational id might be used to store the word pos for instance. Lastly, the key
 * contain the interpretation node of this activation, specifying to which interpretation this activation belongs.
 *
 * <p>The activations are linked to each other on two levels. The fields {@code inputs} and {@code outputs}
 * contain the activation links within the logic layer. The fields {@code inputLinks} and
 * {@code outputLinks} contain the links on the neural layer.
 *
 * @author Lukas Molzberger
 */
public final class Activation extends OrActivation {

    public static int BEGIN = 0;
    public static int END = 1;

    public static final Comparator<Activation> ACTIVATION_ID_COMP = Comparator.comparingInt(act -> act.id);
    public static int MAX_SELF_REFERENCING_DEPTH = 5;
    public static int MAX_PREDECESSOR_DEPTH = 100;
    public static boolean DEBUG_OUTPUT = false;

    public static Activation MIN_ACTIVATION = new Activation(Integer.MIN_VALUE, null, null);
    public static Activation MAX_ACTIVATION = new Activation(Integer.MAX_VALUE, null, null);

    private static final Logger log = LoggerFactory.getLogger(Activation.class);

    public Map<Integer, Position> slots = new TreeMap<>();

    private TreeSet<Link> selectedInputLinks = new TreeSet<>(INPUT_COMP);
    private TreeMap<Link, Link> inputLinks = new TreeMap<>(INPUT_COMP);
    private TreeMap<Link, Link> outputLinks = new TreeMap<>(OUTPUT_COMP);

    public Integer sequence;

    public double upperBound;
    public double lowerBound;

    public State avgState;
    public List<AvgState> searchStates;

    public Rounds rounds = new Rounds();
    public Rounds finalRounds = rounds;

    public boolean ubQueued = false;
    public long markedHasCandidate;

    public long currentStateV;
    public StateChange currentStateChange;
    public long markedDirty;
    public long markedPredecessor;

    public Double targetValue;
    public Double inputValue;

    public Writable extension;


    public Decision inputDecision = Decision.UNKNOWN;
    public Decision decision = Decision.UNKNOWN;
    public Decision finalDecision = Decision.UNKNOWN;
    public Candidate candidate;
    private long visitedState;
    public long markedAncDesc;

    public boolean blocked;


    private List<Activation> conflicts;


    public Activation(int id, Document doc, OrNode n) {
        super(id, doc, n);

        if(doc != null && doc.model.getActivationExtensionFactory() != null) {
            extension = doc.model.getActivationExtensionFactory().createObject();
        }
    }


    public Position getSlot(int slot) {
        return slots.get(slot);
    }


    public void setSlot(int slot, Position pos) {
        slots.put(slot, pos);
    }


    public void setSlots(Map<Integer, Position> slots) {
        for(Map.Entry<Integer, Position> me: slots.entrySet()) {
            setSlot(me.getKey(), me.getValue());
        }
    }


    public Integer length() {
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        for(Position pos: slots.values()) {
            if(pos.getFinalPosition() == null) {
                return null;
            }
            min = Math.min(min, pos.getFinalPosition());
            max = Math.max(max, pos.getFinalPosition());
        }

        if(min > max) return 0;
        return max - min;
    }


    public void setTargetValue(Double targetValue) {
        this.targetValue = targetValue;
    }


    public String getLabel() {
        return getINeuron().label;
    }

    public Type getType() {
        return getINeuron().type;
    }

    public String getText() {
        return doc.getText(getSlot(BEGIN), getSlot(END));
    }


    public INeuron getINeuron() {
        return getNeuron().get(doc);
    }


    public Neuron getNeuron() {
        return node.neuron;
    }


    public void addLink(Linker.Direction dir, Link l) {
        switch(dir) {
            case INPUT:
                outputLinks.put(l, l);
                break;
            case OUTPUT:
                if(l.input.decision == SELECTED) {
                    selectedInputLinks.add(l);
                }
                inputLinks.put(l, l);
                break;
        }
    }


    public Link getLinkBySynapseId(int synapseId) {
        for(Link l: inputLinks.values()) {
            if(!l.passive && l.synapse.id == synapseId) {
                return l;
            }
        }
        return null;
    }


    public SortedSet<Link> getInputLinksOrderedBySynapse() {
        return inputLinks.navigableKeySet();
    }


    public Stream<Link> getInputLinks(boolean includePassive, boolean onlySelected) {
        Stream<Link> s = (onlySelected ? selectedInputLinks : inputLinks.values()).stream();
        return includePassive ? s : s.filter(l -> !l.passive);
    }


    public Stream<Link> getOutputLinks(boolean includePassive) {
        Stream<Link> s = outputLinks.values().stream();
        return includePassive ? s : s.filter(l -> !l.passive);
    }


    public Link getInputLink(Link l) {
        return inputLinks.get(l);
    }


    public Stream<Link> getInputLinksBySynapse(boolean includePassive, Synapse syn) {
        Stream<Link> s = inputLinks.subMap(
                new Link(syn, MIN_ACTIVATION, MIN_ACTIVATION, false),
                new Link(syn, MAX_ACTIVATION, MAX_ACTIVATION, false))
                .values()
                .stream();
        return includePassive ? s : s.filter(l -> !l.passive);
    }


    public Stream<Link> getOutputLinksBySynapse(boolean includePassive, Synapse syn) {
        Stream<Link> s = outputLinks.subMap(
                new Link(syn, MIN_ACTIVATION, MIN_ACTIVATION, false),
                new Link(syn, MAX_ACTIVATION, MAX_ACTIVATION, false))
                .values()
                .stream();
        return includePassive ? s : s.filter(l -> !l.passive);
    }


    public double process(SearchNode sn, int round, long v) {
        double delta = 0.0;
        State s;
        if(inputValue != null) {
            s = new State(inputValue, inputValue, 1.0, 0.0, 0.0, 0, 0.0);
        } else {
            s = computeValueAndWeight(round);
        }

        if (round == 0 || !rounds.get(round).equalsWithWeights(s)) {
            saveOldState(sn.modifiedActs, v);

            State oldState = rounds.get(round);

            boolean propagate = rounds.set(round, s) && (oldState == null || !oldState.equals(s));

            saveNewState();

            if (propagate) {
                if(round > Document.MAX_ROUND) {
                    log.error("Error: Maximum number of rounds reached. The network might be oscillating.");

                    doc.dumpOscillatingActivations();
                    throw new RuntimeException("Maximum number of rounds reached. The network might be oscillating.");
                } else {
                    if(Document.ROUND_LIMIT < 0 || round < Document.ROUND_LIMIT) {
                        doc.vQueue.propagateActivationValue(round, this);
                    }
                }
            }

            if (round == 0) {
                // In case that there is a positive feedback loop.
                doc.vQueue.add(1, this);
            }

            if (rounds.getLastRound() != null && round >= rounds.getLastRound()) { // Consider only the final round.
                delta += s.weight - oldState.weight;
            }
        }
        return delta;
    }


    public State computeValueAndWeight(int round) {
        INeuron n = getINeuron();
        double net = n.biasSum;
        double posNet = n.biasSum;

        int fired = -1;

        long v = doc.visitedCounter++;
        markPredecessor(v, 0);

        Set<Relation.Weight> relationWeights = null;
        if(Model.ENABLE_RELATION_WEIGHTS) {
            relationWeights = new HashSet<>();
        }

        for (InputState is: getInputStates(round, v)) {
            Synapse s = is.l.synapse;
            Activation iAct = is.l.input;

            if (iAct == this) continue;

            double x = Math.min(s.limit, is.s.value) * s.weight;
            if(s.distanceFunction != null) {
                x *= s.distanceFunction.f(iAct, this);
            }
            net += x;
            if(!s.isNegative()) {
                posNet += x;
            }

            if (!s.isRecurrent && !s.isNegative() && net >= 0.0 && fired < 0) {
                fired = iAct.rounds.get(round).fired + 1;
            }

            if(Model.ENABLE_RELATION_WEIGHTS) {
                collectRelationWeights(relationWeights, is, s);
            }
        }

        if(n.passiveInputSynapses != null) {
            for(Synapse s: n.passiveInputSynapses.values()) {
                double x = s.weight * s.input.get(doc).passiveInputFunction.getActivationValue(s, this);

                net += x;
                if(!s.isNegative()) {
                    posNet += x;
                }
            }
        }

        double actValue = n.activationFunction.f(net);
        double posActValue = n.activationFunction.f(posNet);

        double w = Math.min(-n.negRecSum, net);

        // Compute only the recurrent part is above the threshold.
        double newWeight = decision == SELECTED ? Math.max(0.0, w) : 0.0;

        if(decision == SELECTED || ALLOW_WEAK_NEGATIVE_WEIGHTS) {
            return new State(
                    actValue,
                    posActValue,
                    1.0,
                    net,
                    posNet,
                    -1,
                    newWeight
            );
        } else {
            return new State(
                    0.0,
                    posActValue,
                    0.0,
                    0.0,
                    posNet,
                    -1,
                    newWeight
            );
        }
    }

    public void collectRelationWeights(Set<Relation.Weight> relationWeights, InputState is, Synapse s) {
        for(Map.Entry<Integer, Relation> me: s.relations.entrySet()) {
            Integer relId = me.getKey();
            Relation r = me.getValue();

            Activation rAct = null;
            if(relId == Synapse.OUTPUT) {
                rAct = this;
            } else {
                Link rl = getLinkBySynapseId(relId);
                if(rl != null) {
                    rAct = rl.input;
                }
            }

            if(rAct != null && r.test(is.l.input, rAct)) {
                relationWeights.add(r.getWeight());
            }
        }
    }


    public boolean isActiveable() {
        INeuron n = getINeuron();
        double net = n.biasSum;

        for (Link l: inputLinks.values()) {
            if(l.synapse.inactive || l.passive) {
                continue;
            }

            Synapse s = l.synapse;
            Activation iAct = l.input;

            if (iAct == this) continue;

            double iv = 0.0;
            if(!l.synapse.isNegative() && l.input.decision != EXCLUDED) {
                iv = Math.min(l.synapse.limit, l.input.upperBound);
            }

            double x = iv * s.weight;
            if(s.distanceFunction != null) {
                x *= s.distanceFunction.f(iAct, this);
            }
            net += x;
        }

        if(n.passiveInputSynapses != null) {
            for(Synapse s: n.passiveInputSynapses.values()) {
                double x = s.weight * s.input.get(doc).passiveInputFunction.getActivationValue(s, this);

                net += x;
            }
        }

        return net > 0.0;
    }


    public void processBounds() {
        double oldUpperBound = upperBound;

        computeBounds();

        if(Math.abs(upperBound - oldUpperBound) > 0.01) {
            for(Link l: outputLinks.values()) {
                if(!l.passive) {
                    doc.ubQueue.add(l);
                }
            }
        }

        if (oldUpperBound <= 0.0 && upperBound > 0.0 && !blocked) {
            getINeuron().propagate(this);
        }
    }


    public void computeBounds() {
        INeuron n = getINeuron();
        double ub = n.biasSum + n.posRecSum;
        double lb = n.biasSum + n.posRecSum;

        long v = doc.visitedCounter++;
        markPredecessor(v, 0);

        for (Link l : inputLinks.values()) {
            Synapse s = l.synapse;
            if(s.inactive || l.passive) {
                continue;
            }

            Activation iAct = l.input;

            if (iAct == this) continue;

            double x = s.weight;
            if(s.distanceFunction != null) {
                x *= s.distanceFunction.f(iAct, this);
            }

            if (s.isNegative()) {
                if (!s.isRecurrent && !iAct.checkSelfReferencing(false, 0, v)) {
                    ub += Math.min(s.limit, iAct.lowerBound) * x;
                }

                lb += s.limit * x;
            } else {
                ub += Math.min(s.limit, iAct.upperBound) * x;
                lb += Math.min(s.limit, iAct.lowerBound) * x;
            }
        }

        if(n.passiveInputSynapses != null) {
            for(Synapse s: n.passiveInputSynapses.values()) {
                double x = s.weight * s.input.get(doc).passiveInputFunction.getActivationValue(s, this);

                ub += x;
                lb += x;
            }
        }

        upperBound = n.activationFunction.f(ub);
        lowerBound = n.activationFunction.f(lb);
    }


    private static State getInitialState(Decision c) {
        return new State(
                c == SELECTED ? 1.0 : 0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0,
                0.0
        );
    }



    private List<InputState> getInputStates(int round, long v) {
        ArrayList<InputState> tmp = new ArrayList<>();
        Synapse lastSynapse = null;
        InputState maxInputState = null;
        for (Link l : inputLinks.values()) {
            if(l.synapse.inactive || l.passive) {
                continue;
            }
            if (lastSynapse != null && lastSynapse != l.synapse) {
                tmp.add(maxInputState);
                maxInputState = null;
            }

            State s = l.input.getInputState(round, l.synapse, v);
            if (maxInputState == null || maxInputState.s.value < s.value) {
                maxInputState = new InputState(l, s);
            }
            lastSynapse = l.synapse;
        }
        if (maxInputState != null) {
            tmp.add(maxInputState);
        }

        return tmp;
    }


    /*
    An activable activation object might still be suppressed by an undecided positive feedback link.
     */
    public boolean hasUndecidedPositiveFeedbackLinks() {
        return getInputLinks(false, false)
                .anyMatch(l -> l.synapse.isRecurrent && !l.synapse.isNegative() && l.input.decision == UNKNOWN);
    }


    private static class InputState {
        public InputState(Link l, State s) {
            this.l = l;
            this.s = s;
        }

        Link l;
        State s;
    }


    private State getInputState(int round, Synapse s, long v) {
        State is = State.ZERO;
        if (s.isRecurrent) {
            if (!s.isNegative() || !checkSelfReferencing(true, 0, v)) {
                is = round == 0 ? getInitialState(decision) : rounds.get(round - 1);
            }
        } else {
            is = rounds.get(round);
        }
        return is;
    }


    public List<Link> getFinalInputActivationLinks() {
        ArrayList<Link> results = new ArrayList<>();
        for (Link l : inputLinks.values()) {
            if (!l.passive && l.input.isFinalActivation()) {
                results.add(l);
            }
        }
        return results;
    }


    public List<Link> getFinalOutputActivationLinks() {
        ArrayList<Link> results = new ArrayList<>();
        for (Link l : outputLinks.values()) {
            if (!l.passive && l.output.isFinalActivation()) {
                results.add(l);
            }
        }
        return results;
    }


    public Collection<Activation> getConflicts() {
        if(conflicts != null) {
            return conflicts;
        }

        long v = doc.visitedCounter++;
        markPredecessor(v, 0);
        conflicts = new ArrayList<>();
        for(Link l: inputLinks.values()) {
            if (!l.passive && l.synapse.isNegative() && l.synapse.isRecurrent) {
                l.input.collectIncomingConflicts(conflicts, v);
            }
        }
        collectOutgoingConflicts(conflicts, v);
        return conflicts;
    }


    private void collectIncomingConflicts(List<Activation> conflicts, long v) {
        if(markedPredecessor == v) return;

        if (getINeuron().type != INeuron.Type.INHIBITORY) {
            conflicts.add(this);
        } else {
            for (Link l : inputLinks.values()) {
                if (!l.passive && !l.synapse.isNegative() && !l.synapse.isRecurrent) {
                    l.input.collectIncomingConflicts(conflicts, v);
                }
            }
        }
    }


    private void collectOutgoingConflicts(List<Activation> conflicts, long v) {
        if(markedPredecessor == v) return;

        for(Link l: outputLinks.values()) {
            if(l.passive) {
                continue;
            }
            if (l.output.getINeuron().type != INeuron.Type.INHIBITORY) {
                if (l.synapse.isNegative() && l.synapse.isRecurrent) {
                    conflicts.add(l.output);
                }
            } else if (!l.synapse.isNegative() && !l.synapse.isRecurrent) {
                l.output.collectOutgoingConflicts(conflicts, v);
            }
        }
    }


    public void adjustSelectedNeuronInputs(Decision d) {
        for(Link l: outputLinks.values()) {
            if(l.passive) {
                continue;
            }
            if(d == SELECTED) {
                l.output.selectedInputLinks.add(l);
            } else {
                l.output.selectedInputLinks.remove(l);
            }
        }
    }


    public boolean checkSelfReferencing(boolean onlySelected, int depth, long v) {
        if (markedPredecessor == v) {
            return true;
        }

        if (depth > MAX_SELF_REFERENCING_DEPTH) {
            return false;
        }

        for (Link l: onlySelected ? selectedInputLinks : inputLinks.values()) {
            if(!l.passive && !l.synapse.isNegative()) {
                if (l.input.checkSelfReferencing(onlySelected, depth + 1, v)) {
                    return true;
                }
            }
        }

        return false;
    }


    public void setDecision(Decision newDecision, long v) {
        if(inputDecision != Decision.UNKNOWN && newDecision != inputDecision) return;

        if (newDecision == Decision.UNKNOWN && v != visitedState) return;

        if(decision == Decision.SELECTED != (newDecision == Decision.SELECTED)) {
            adjustSelectedNeuronInputs(newDecision);
        }

        decision = newDecision;
        visitedState = v;
    }


    public boolean isFinalActivation() {
        return getFinalState().value > 0.0;
    }


    public State getFinalState() {
        return finalRounds.getLast();
    }


    public Integer getSequence() {
        if (sequence != null) return sequence;

        sequence = 0;
        inputLinks
                .values()
                .stream()
                .filter(l -> !l.synapse.isRecurrent && !l.passive)
                .forEach(l -> sequence = Math.max(sequence, l.input.getSequence() + 1));
        return sequence;
    }


    public void markDirty(long v) {
        markedDirty = Math.max(markedDirty, v);
    }


    public void markPredecessor(long v, int depth) {
        if(depth > MAX_PREDECESSOR_DEPTH) {
            throw new RuntimeException("MAX_PREDECESSOR_DEPTH limit exceeded. Probable cause is a non recurrent loop.");
        }

        markedPredecessor = v;

        for(Link l: inputLinks.values()) {
            if(!l.passive && !l.synapse.isNegative() && !l.synapse.isRecurrent) {
                l.input.markPredecessor(v, depth + 1);
            }
        }
    }


    /**
     * Since Aika is a recurrent neural network, it is necessary to compute several rounds of activation values. The
     * computation stops if no further changes occur to the state. Only the recurrent synapses depend on the previous
     * round.
     *
     */
    public static class Rounds {
        private boolean[] isQueued = new boolean[3];

        public TreeMap<Integer, State> rounds = new TreeMap<>();


        public Rounds() {
            rounds.put(0, State.ZERO);
        }


        public boolean set(int r, State s) {
            State lr = get(r - 1);
            if(lr != null && lr.equalsWithWeights(s)) {
                State or = rounds.get(r);
                if(or != null) {
                    rounds.remove(r);
                    return !or.equalsWithWeights(s);
                }
                return false;
            } else {
                State or = rounds.put(r, s);

                for(Iterator<Map.Entry<Integer, State>> it = rounds.tailMap(r + 1).entrySet().iterator(); it.hasNext(); ) {
                    Map.Entry<Integer, State> me = it.next();
                    if(me.getValue().equalsWithWeights(s)) it.remove();
                }
                return or == null || !or.equalsWithWeights(s);
            }
        }


        public State get(int r) {
            Map.Entry<Integer, State> me = rounds.floorEntry(r);
            return me != null ? me.getValue() : null;
        }

        public Rounds copy() {
            Rounds nr = new Rounds();
            nr.rounds.putAll(rounds);
            return nr;
        }

        public Integer getLastRound() {
            return !rounds.isEmpty() ? rounds.lastKey() : null;
        }

        public State getLast() {
            return !rounds.isEmpty() ? rounds.lastEntry().getValue() : State.ZERO;
        }

        public void setQueued(int r, boolean v) {
            if(r >= isQueued.length) {
                isQueued = Arrays.copyOf(isQueued, isQueued.length * 2);
            }
            isQueued[r] = v;
        }

        public boolean isQueued(int r) {
            return r < isQueued.length ? isQueued[r] : false;
        }


        public void reset() {
            rounds.clear();
            rounds.put(0, State.ZERO);
        }

        public String toString() {
            StringBuilder sb = new StringBuilder();
            rounds.forEach((r, s) -> sb.append(r + ":" + s.value + " "));
            return sb.toString();
        }


        public boolean compare(Rounds r) {
            if(rounds.size() != r.rounds.size()) {
                return false;
            }
            for(Map.Entry<Integer, State> me: rounds.entrySet()) {
                State sa = me.getValue();
                State sb = r.rounds.get(me.getKey());
                if(sb == null || Math.abs(sa.value - sb.value) > 0.0000001) {
                    return false;
                }
            }

            return true;
        }


        public boolean isActive() {
            return rounds.size() <= 1 && getLast().value > 0.0;
        }
    }


    /**
     * A <code>State</code> object contains the activation value of an activation object that belongs to a neuron.
     * It furthermore contains a weight that is used to check the interpretations during the search for the best
     * interpretation.
     */
    public static class State {
        public final double value;
        public final double posValue;
        public final double p;
        public final double net;
        public final double posNet;

        public final int fired;
        public final double weight;

        public static final State ZERO = new State(0.0, 0.0, 0.0, 0.0, 0.0, -1, 0.0);

        public State(double value, double posValue, double p, double net, double posNet, int fired, double weight) {
            assert !Double.isNaN(value);
            this.value = value;
            this.posValue = posValue;
            this.p = p;
            this.net = net;
            this.posNet = posNet;
            this.fired = fired;
            this.weight = weight;
        }


        public boolean equals(State s) {
            return Math.abs(value - s.value) <= INeuron.WEIGHT_TOLERANCE;
        }

        public boolean equalsWithWeights(State s) {
            return equals(s) && Math.abs(weight - s.weight) <= INeuron.WEIGHT_TOLERANCE;
        }

        public String toString() {
            return "V:" + Utils.round(value) + (DEBUG_OUTPUT ? " pV:" + Utils.round(posValue) : "") + " Net:" + Utils.round(net) + (DEBUG_OUTPUT ? " P:" + Utils.round(p) : "") + " W:" + Utils.round(weight);
        }
    }


    public String toString() {
        return id + " " + slotsToString() + " " + identityToString() + " - " + node + " -" +
                (extension != null ? extension.toString() + " -" : "") +
                " UB:" + Utils.round(upperBound) +
                (inputValue != null ? " IV:" + Utils.round(inputValue) : "") +
                (targetValue != null ? " TV:" + Utils.round(targetValue) : "") +
                " V:" + Utils.round(rounds.getLast().value) +
                " FV:" + Utils.round(finalRounds.getLast().value);
    }



    public String toString(boolean withLogic) {
        StringBuilder sb = new StringBuilder();
        sb.append(id + " - ");

        sb.append(finalDecision + " - ");

        sb.append(slotsToString());

        sb.append(" \"");
        if (node.neuron.get().getOutputText() != null) {
            sb.append(Utils.collapseText(node.neuron.get().getOutputText(), 7));
        } else {
            sb.append(Utils.collapseText(doc.getText(getSlot(BEGIN), getSlot(END)), 7));
        }
        sb.append("\"");

        sb.append(identityToString());
        sb.append(" - ");

        sb.append(withLogic ? node.toString() : node.getNeuronLabel());

        if(extension != null) {
            sb.append(" - " + extension);
        }

        if(DEBUG_OUTPUT) {
            sb.append(" - UB:");
            sb.append(Utils.round(upperBound));
        }

        if(SearchNode.COMPUTE_SOFT_MAX && avgState != null) {
            sb.append(" AVG:");
            sb.append(avgState);
        }

        sb.append(" - ");
        if (isFinalActivation()) {
            State fs = getFinalState();
            sb.append(fs);
        }


        if (inputValue != null) {
            sb.append(" - IV:" + Utils.round(inputValue));
        }

        if (targetValue != null) {
            sb.append(" - TV:" + Utils.round(targetValue));
        }

        return sb.toString();
    }


    public String slotsToString() {
        StringBuilder sb = new StringBuilder();
        sb.append("(");
        boolean first = true;
        for(Map.Entry<Integer, Position> me: slots.entrySet()) {
            if(!first) {
                sb.append(", ");
            }
            first = false;

            sb.append(me.getKey());
            sb.append(":");
            sb.append(me.getValue());
        }
        sb.append(")");
        return sb.toString();
    }


    public String identityToString() {
        StringBuilder sb = new StringBuilder();
        sb.append(" (");
        boolean first = true;
        for(Link l: inputLinks.values()) {
            if(!l.passive && l.synapse.identity) {
                if(!first) {
                    sb.append(", ");
                }

                sb.append(l.input.id);

                first = false;
            }
        }
        sb.append(")");
        return sb.toString();
    }


    public String linksToString() {
        StringBuilder sb = new StringBuilder();
        for(Link l: inputLinks.values()) {
            if(!l.passive) {
                sb.append("  " + l.input.getLabel() + "  W:" + l.synapse.weight + "\n");
            }
        }

        return sb.toString();
    }


    public enum Mode {OLD, NEW}

    public void saveOldState(Map<Activation, StateChange> changes, long v) {
        StateChange sc = currentStateChange;
        if (sc == null || currentStateV != v) {
            sc = new StateChange();
            sc.oldRounds = rounds.copy();
            currentStateChange = sc;
            currentStateV = v;
            if (changes != null) {
                changes.put(sc.getActivation(), sc);
            }
        }
    }

    public void saveNewState() {
        StateChange sc = currentStateChange;

        sc.newRounds = rounds.copy();
        sc.newState = decision;
    }


    /**
     * The {@code StateChange} class is used to store the state change of an activation that occurs in each node of
     * the binary search tree. When a candidate refinement is selected during the search, then the activation values of
     * all affected activation objects are adjusted. The changes to the activation values are also propagated through
     * the network. The old state needs to be stored here in order for the search to be able to restore the old network
     * state before following the alternative search branch.
     */
    public class StateChange {
        public Rounds oldRounds;
        public Rounds newRounds;
        public Decision newState;

        public void restoreState(Mode m) {
            rounds = (m == Mode.OLD ? oldRounds : newRounds).copy();
        }

        public Activation getActivation() {
            return Activation.this;
        }
    }


    public static class AvgState {
        public State state;
        public double weight;

        public AvgState(State state, double weight) {
            this.state = state;
            this.weight = weight;
        }
    }


    /**
     * The {@code SynapseActivation} mirror the synapse link in the network of activations.
     */
    public static class Link {
        public final Synapse synapse;
        public final Activation input;
        public final Activation output;
        public boolean passive;

        public static Comparator<Link> INPUT_COMP = (l1, l2) -> {
            int r = Synapse.INPUT_SYNAPSE_COMP.compare(l1.synapse, l2.synapse);
            if (r != 0) return r;
            return Integer.compare(l1.input.id, l2.input.id);
        };

        public static Comparator<Link> OUTPUT_COMP = (l1, l2) -> {
            int r = Synapse.OUTPUT_SYNAPSE_COMP.compare(l1.synapse, l2.synapse);
            if (r != 0) return r;
            return Integer.compare(l1.output.id, l2.output.id);
        };


        public Link(Synapse s, Activation input, Activation output, boolean passive) {
            this.synapse = s;
            this.input = input;
            this.output = output;
            this.passive = passive;
        }


        public void link() {
            input.addLink(INPUT, this);
            output.addLink(OUTPUT, this);
        }

        public String toString() {
            return (passive ? "p" : "") + synapse + ": " + input + " --> " + output;
        }
    }

    public static class Builder {
        public SortedMap<Integer, Integer> positions = new TreeMap<>();
        public double value = 1.0;
        public double net = 0.0;
        public Double targetValue;
        public int fired;


        public Builder setRange(int begin, int end) {
            setPosition(Activation.BEGIN, begin);
            setPosition(Activation.END, end);
            return this;
        }

        public Builder setPosition(int slot, int pos) {
            positions.put(slot, pos);
            return this;
        }

        public Builder setValue(double value) {
            this.value = value;
            return this;
        }

        public Builder setNet(double net) {
            this.net = net;
            return this;
        }

        public Builder setTargetValue(Double targetValue) {
            this.targetValue = targetValue;
            return this;
        }

        public Builder setFired(int fired) {
            this.fired = fired;
            return this;
        }
    }
}

