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
import network.aika.Document;
import network.aika.neuron.relation.Relation;
import network.aika.neuron.INeuron.SynapseSummary;
import network.aika.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.*;

import static network.aika.neuron.Synapse.State.CURRENT;
import static network.aika.neuron.Synapse.State.NEXT;

/**
 * The {@code Synapse} class connects two neurons with each other. When propagating an activation signal, the
 * weight of the synapse is multiplied with the activation value of the input neurons activation. The result is then added
 * to the output neurons weighted sum to compute the output activation value. In contrast to a conventional neural network
 * synapses in Aika do not just propagate the activation value from one neuron to the next, but also structural information
 * like the text range and the relational id (e.g. word position) and also the interpretation to which the input
 * activation belongs. To determine in which way the structural information should be propagated synapses in Aika possess
 * a few more properties.
 *
 * <p>The properties {@code relativeRid} and {@code absoluteRid} determine either the relative difference between
 * the {@code rid} of the input activation and the rid of the output activation or require a fixed rid as input.
 *
 * <p>The properties range match, range mapping and range output manipulate the ranges. The range match determines whether
 * the input range begin or end is required to be equal, greater than or less than the range of the output activation.
 * The range mapping can be used to map for example an input range end to an output range begin. Usually this simply maps
 * begin to begin and end to end. The range output property is a boolean flag which determines whether the input range
 * should be propagated to the output activation.
 *
 * <p>Furthermore, the property {@code isRecurrent} specifies if this synapse is a recurrent feedback link. Recurrent
 * feedback links can be either negative or positive depending on the weight of the synapse. Recurrent feedback links
 * kind of allow to use future information as inputs of a current neuron. Aika allows this by making assumptions about
 * the recurrent input neuron. The class {@code SearchNode} modifies these assumptions until the best interpretation
 * for this document is found.
 *
 *
 * @author Lukas Molzberger
 */
public class Synapse implements Writable {

    public static final int OUTPUT = -1;

    public static double DISJUNCTION_THRESHOLD = 0.6;
    public static double CONJUNCTION_THRESHOLD = 0.4;

    public static double TOLERANCE = 0.0000001;


    public static final Comparator<Synapse> INPUT_SYNAPSE_COMP = (s1, s2) -> {
        int r = s1.input.compareTo(s2.input);
        if (r != 0) return r;
        return Integer.compare(s1.id, s2.id);
    };


    public static final Comparator<Synapse> OUTPUT_SYNAPSE_COMP = (s1, s2) -> {
        int r = s1.output.compareTo(s2.output);
        if (r != 0) return r;
        return Integer.compare(s1.id, s2.id);
    };

    private Neuron input;
    private Neuron output;

    private Integer id;

    private boolean isRecurrent;
    private boolean identity;

    private Map<Integer, Relation> relations = new TreeMap<>();

    private DistanceFunction distanceFunction = null;

    private Writable extension;


    private boolean inactive;

    private double weight;
    private double weightDelta;

    private double bias;
    private double biasDelta;

    private double limit = 1.0;
    private double limitDelta;

    private boolean isConjunction;
    private boolean isDisjunction;


    public Synapse() {
    }


    public Synapse(Neuron input, Neuron output, Integer id) {
        this.id = id;
        this.input = input;
        this.output = output;

        if(output.getModel().getSynapseExtensionFactory() != null) {
            extension = output.getModel().getSynapseExtensionFactory().createObject();
        }
    }


    public Neuron getInput() {
        return input;
    }

    public Neuron getOutput() {
        return output;
    }

    public Integer getId() {
        return id;
    }

    public boolean isRecurrent() {
        return isRecurrent;
    }

    public void setRecurrent(boolean recurrent) {
        isRecurrent = recurrent;
    }

    public boolean isIdentity() {
        return identity;
    }

    public void setIdentity(boolean identity) {
        this.identity = identity;
    }

    public Map<Integer, Relation> getRelations() {
        return relations;
    }

    public void setRelations(Map<Integer, Relation> relations) {
        this.relations = relations;
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }

    public void setDistanceFunction(DistanceFunction distanceFunction) {
        this.distanceFunction = distanceFunction;
    }

    public <T extends Writable> T getExtension() {
        return (T) extension;
    }

    public void setExtension(Writable extension) {
        this.extension = extension;
    }

    public boolean isInactive() {
        return inactive;
    }

    public void setInactive(boolean inactive) {
        this.inactive = inactive;
    }

    public boolean isConjunction() {
        return isConjunction;
    }

    public boolean isDisjunction() {
        return isDisjunction;
    }


    public double getWeight() {
        return weight;
    }

    public double getBias() {
        return bias;
    }

    public double getLimit() {
        return limit;
    }

    public double getNewWeight() {
        return weight + weightDelta;
    }

    public double getNewBias() {
        return bias + biasDelta;
    }

    public double getNewLimit() {
        return limit + limitDelta;
    }

    public double getWeight(State s) {
        return s == State.CURRENT ? weight : getNewWeight();
    }

    public double getBias(State s) {
        return s == State.CURRENT ? bias : getNewBias();
    }

    public double getLimit(State s) {
        return s == State.CURRENT ? limit : getNewLimit();
    }

    public double getWeightDelta() {
        return weightDelta;
    }

    public double getBiasDelta() {
        return biasDelta;
    }

    public double getLimitDelta() {
        return limitDelta;
    }

    public void link() {
        INeuron in = input.get();
        INeuron out = output.get();

        boolean dir = in.getId() < out.getId();

        (dir ? in : out).lock.acquireWriteLock();
        (dir ? out : in).lock.acquireWriteLock();

        input.lock.acquireWriteLock();
        input.activeOutputSynapses.put(this, this);
        input.lock.releaseWriteLock();

        output.lock.acquireWriteLock();
        output.activeInputSynapses.put(this, this);
        output.inputSynapsesById.put(id, this);
        output.lock.releaseWriteLock();

        removeLinkInternal(in, out);

        isConjunction = isConjunction(NEXT);
        if(isConjunction) {
            out.inputSynapses.put(this, this);
            out.setModified();
        }

        isDisjunction = isDisjunction(State.NEXT);
        if(isDisjunction){
            in.outputSynapses.put(this, this);
            in.setModified();
        }

        out.registerSynapseId(id);

        if(in.isPassiveInputNeuron()) {
            out.registerPassiveInputSynapse(this);
        }

        (dir ? in : out).lock.releaseWriteLock();
        (dir ? out : in).lock.releaseWriteLock();
    }


    public void relink() {
        boolean newIsConjunction = isConjunction(NEXT);
        if(newIsConjunction != isConjunction) {
            INeuron in = input.get();
            INeuron out = output.get();

            boolean dir = in.getId() < out.getId();
            (dir ? in : out).lock.acquireWriteLock();
            (dir ? out : in).lock.acquireWriteLock();

            isConjunction = newIsConjunction;
            if (isConjunction) {
                out.inputSynapses.put(this, this);
                out.setModified();
            } else {
                out.inputSynapses.remove(this);
                out.setModified();
            }

            (dir ? in : out).lock.releaseWriteLock();
            (dir ? out : in).lock.releaseWriteLock();
        }

        boolean newIsDisjunction = isDisjunction(NEXT);
        if(newIsDisjunction != isDisjunction) {
            INeuron in = input.get();
            INeuron out = output.get();

            boolean dir = in.getId() < out.getId();
            (dir ? in : out).lock.acquireWriteLock();
            (dir ? out : in).lock.acquireWriteLock();

            isDisjunction = newIsDisjunction;
            if (isDisjunction) {
                in.outputSynapses.put(this, this);
                in.setModified();
            } else {
                in.outputSynapses.remove(this);
                in.setModified();
            }

            (dir ? in : out).lock.releaseWriteLock();
            (dir ? out : in).lock.releaseWriteLock();
        }
    }


    public void unlink() {
        INeuron in = input.get();
        INeuron out = output.get();

        boolean dir = input.getId() < out.getId();

        (dir ? in : out).lock.acquireWriteLock();
        (dir ? out : in).lock.acquireWriteLock();

        input.lock.acquireWriteLock();
        input.activeOutputSynapses.remove(this);
        input.lock.releaseWriteLock();

        output.lock.acquireWriteLock();
        output.activeInputSynapses.remove(this);
        output.inputSynapsesById.remove(id);
        output.lock.releaseWriteLock();

        removeLinkInternal(in, out);

        (dir ? in : out).lock.releaseWriteLock();
        (dir ? out : in).lock.releaseWriteLock();
    }


    private void removeLinkInternal(INeuron in, INeuron out) {
        if(isConjunction(CURRENT)) {
            if(out.inputSynapses.remove(this) != null) {
                out.setModified();
            }
        }
        if(isDisjunction(CURRENT)) {
            if(in.outputSynapses.remove(this) != null) {
                in.setModified();
            }
        }
    }


    public double getMaxInputValue() {
        return limit * weight;
    }


    public boolean exists() {
        if(input.get().outputSynapses.containsKey(this)) return true;
        if(output.get().inputSynapses.containsKey(this)) return true;
        return false;
    }


    public void commit() {
        weight += weightDelta;
        weightDelta = 0.0;

        bias += biasDelta;
        biasDelta = 0.0;

        limit += limitDelta;
        limitDelta = 0.0;
    }


    public boolean isZero() {
        return Math.abs(weight) < TOLERANCE && Math.abs(bias) < TOLERANCE ;
    }


    public enum State {
        NEXT,
        CURRENT
    }


    public boolean isConjunction(State state) {
        double w = getLimit(state) * getWeight(state);
        double b = getBias(state);
        return w < 0.0 || (w > 0.0 && (-b / w) >= CONJUNCTION_THRESHOLD);
    }


    public boolean isDisjunction(State state) {
        double w = getLimit(state) * getWeight(state);
        double b = getBias(state);
        return w > 0.0 && (-b / w) <= DISJUNCTION_THRESHOLD;
    }


    public boolean isWeak(State state) {
        double w = getLimit(state) * getWeight(state);

        SynapseSummary ss = output.get().getSynapseSummary();
        if(w > -ss.getBiasSum(state)) {
            return false;
        }

        if(w > ss.getBiasSum(state) + ss.getPosSum(state)) {
            return false;
        }
        return true;
    }


    public void updateDelta(Document doc, double weightDelta, double biasDelta, double limitDelta) {
        this.weightDelta += weightDelta;
        this.biasDelta += biasDelta;
        this.limitDelta += limitDelta;
        relink();
        if(doc != null) {
            doc.notifyWeightModified(this);
        }
    }


    public void update(Document doc, double weight, double bias, double limit) {
        this.weightDelta = weight - this.weight;
        this.limitDelta = limit - this.limit;
        this.biasDelta = bias - this.bias;

        relink();
        if(doc != null) {
            doc.notifyWeightModified(this);
        }
    }


    public boolean isNegative(State s) {
        return getWeight(s) < 0.0;
    }


    public Relation getRelationById(Integer id) {
        return relations.get(id);
    }


    public String toString() {
        return "S ID:" + id + " NW:" + getNewWeight() + " NB:" + getNewBias() + " rec:" + isRecurrent + " " +  input + "->" + output;
    }


    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(id);

        out.writeBoolean(isRecurrent);
        out.writeBoolean(identity);

        out.writeInt(input.getId());
        out.writeInt(output.getId());

        out.writeInt(relations.size());
        for(Map.Entry<Integer, Relation> me: relations.entrySet()) {
            out.writeInt(me.getKey());

            me.getValue().write(out);
        }

        out.writeBoolean(distanceFunction != null);
        if(distanceFunction != null) {
            out.writeUTF(distanceFunction.name());
        }

        out.writeDouble(weight);
        out.writeDouble(bias);
        out.writeDouble(limit);

        out.writeBoolean(isConjunction);
        out.writeBoolean(isDisjunction);

        out.writeBoolean(inactive);

        out.writeBoolean(extension != null);
        if(extension != null) {
            extension.write(out);
        }
    }


    @Override
    public void readFields(DataInput in, Model m) throws IOException {
        id = in.readInt();

        isRecurrent = in.readBoolean();
        identity = in.readBoolean();

        input = m.lookupNeuron(in.readInt());
        output = m.lookupNeuron(in.readInt());

        int l = in.readInt();
        for(int i = 0; i < l; i++) {
            Integer relId = in.readInt();
            relations.put(relId, Relation.read(in, m));
        }

        if(in.readBoolean()) {
            distanceFunction = DistanceFunction.valueOf(in.readUTF());
        }

        weight = in.readDouble();
        bias = in.readDouble();
        limit = in.readDouble();

        isConjunction = in.readBoolean();
        isDisjunction = in.readBoolean();

        inactive = in.readBoolean();

        if(in.readBoolean()) {
            extension = m.getSynapseExtensionFactory().createObject();
            extension.readFields(in, m);
        }
    }


    public static Synapse read(DataInput in, Model m) throws IOException {
        Synapse s = new Synapse();
        s.readFields(in, m);
        return s;
    }



    public static Synapse createOrReplace(Document doc, Integer synapseId, Neuron inputNeuron, Neuron outputNeuron) {
        outputNeuron.get(doc);
        inputNeuron.get(doc);
        outputNeuron.lock.acquireWriteLock();
        Synapse synapse = outputNeuron.inputSynapsesById.get(synapseId);
        if(synapse != null) {
            synapse.unlink();
        }
        outputNeuron.lock.releaseWriteLock();

        if(synapse == null) {
            synapse = new Synapse(inputNeuron, outputNeuron, synapseId);
        } else {
            synapse.input = inputNeuron;
            synapse.output = outputNeuron;
        }

        if (synapseId == null) {
            synapse.id = outputNeuron.get(doc).getNewSynapseId();
        }

        synapse.link();

        return synapse;
    }

    public Set<Integer> linksOutput() {
        Set<Integer> results = new TreeSet<>();
        for(Map.Entry<Integer, Relation> me: relations.entrySet()) {
            Relation rel = me.getValue();
            if(me.getKey() == OUTPUT) {
                rel.linksOutputs(results);
            }
        }
        return results;
    }


    public boolean linksAnyOutput() {
        Set<Integer> tmp = linksOutput();
        return !tmp.isEmpty();
    }


    /**
     * The {@code Builder} class is just a helper class which is used to initialize a neuron. Most of the parameters of this class
     * will be mapped to a input synapse for this neuron.
     *
     * @author Lukas Molzberger
     */
    public static class Builder implements Neuron.Builder {

        public boolean recurrent;
        public Neuron neuron;
        public double weight;
        public double bias;
        public double limit = 1.0;

        public DistanceFunction distanceFunction;

        public boolean identity;

        public Integer synapseId;


        /**
         * The property <code>recurrent</code> specifies if input is a recurrent feedback link. Recurrent
         * feedback links can be either negative or positive depending on the weight of the synapse. Recurrent feedback links
         * kind of allow to use future information as inputs of a current neuron. Aika allows this by making assumptions about
         * the recurrent input neuron. The class <code>SearchNode</code> modifies these assumptions until the best interpretation
         * for this document is found.
         *
         * @param recurrent
         * @return
         */
        public Builder setRecurrent(boolean recurrent) {
            this.recurrent = recurrent;
            return this;
        }


        /**
         * Determines the input neuron.
         *
         * @param neuron
         * @return
         */
        public Builder setNeuron(Neuron neuron) {
            assert neuron != null;
            this.neuron = neuron;
            return this;
        }


        /**
         * The synapse weight of this input.
         *
         * @param weight
         * @return
         */
        public Builder setWeight(double weight) {
            this.weight = weight;
            return this;
        }

        /**
         * The bias of this input that will later on be added to the neurons bias.
         *
         * @param bias
         * @return
         */
        public Builder setBias(double bias) {
            this.bias = bias;
            return this;
        }


        public Builder setLimit(double limit) {
            this.limit = limit;
            return this;
        }


        public Builder setDistanceFunction(DistanceFunction distFunc) {
            this.distanceFunction = distFunc;
            return this;
        }




        public Builder setIdentity(boolean identity) {
            this.identity = identity;
            return this;
        }


        public Builder setSynapseId(int synapseId) {
            assert synapseId >= 0;
            this.synapseId = synapseId;
            return this;
        }


        @Override
        public void registerSynapseIds(Neuron n) {
            n.registerSynapseId(synapseId);
        }


        public Synapse getSynapse(Neuron outputNeuron) {
            Synapse s = createOrReplace(null, synapseId, neuron, outputNeuron);

            s.isRecurrent = recurrent;
            s.identity = identity;
            s.distanceFunction = distanceFunction;

            return s;
        }
    }
}
