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
package org.aika;

import org.aika.corpus.Range.Operator;
import org.aika.corpus.Range.Mapping;
import org.aika.neuron.Neuron;


/**
 *
 * @author Lukas Molzberger
 */
public class Input implements Comparable<Input> {
    public boolean recurrent;
    public boolean optional;
    public Neuron neuron;
    public double weight;
    public double maxLowerWeightsSum = Double.MAX_VALUE;
    public double minInput;
    public Operator startRangeMatch = Operator.NONE;
    public Operator endRangeMatch = Operator.NONE;
    public boolean startRangeOutput;
    public boolean endRangeOutput;
    public Mapping startMapping = Mapping.START;
    public Mapping endMapping = Mapping.END;

    public Integer relativeRid;
    public Integer absoluteRid;


    /**
     * If recurrent is set to true, then this input will describe an feedback loop.
     * The input neuron may depend on the output of this neuron.
     *
     * @param recurrent
     * @return
     */
    public Input setRecurrent(boolean recurrent) {
        this.recurrent = recurrent;
        return this;
    }

    /**
     * If optional is set to true, then this input is an optional part of a conjunction.
     * This is only used for the method createAndNeuron.
     *
     * @param optional
     * @return
     */
    public Input setOptional(boolean optional) {
        this.optional = optional;
        return this;
    }

    /**
     * Determines the input neuron.
     *
     * @param neuron
     * @return
     */
    public Input setNeuron(Neuron neuron) {
        assert neuron != null;
        this.neuron = neuron;
        return this;
    }

    /**
     * MaxLowerWeightsSum is the expected sum of all weights smaller then the current weight. It is
     * used as an hint to compute the boolean representation of this neuron.
     *
     * @param maxLowerWeightsSum
     * @return
     */
    public Input setMaxLowerWeightsSum(double maxLowerWeightsSum) {
        this.maxLowerWeightsSum = maxLowerWeightsSum;
        return this;
    }

    /**
     * The synapse weight of this input.
     *
     * @param weight
     * @return
     */
    public Input setWeight(Double weight) {
        this.weight = weight;
        return this;
    }

    /**
     * The minimum activation value that is required for this input. The minInput
     * value is used to compute the neurons bias. It is only applied in the createAndNeuron method and only affects inputs with a positive weight and the optional flag set to false.
     *
     * @param minInput
     * @return
     */
    public Input setMinInput(double minInput) {
        this.minInput = minInput;
        return this;
    }

    /**
     * If the absolute relational id (rid) not null, then it is required to match the rid of input activation.
     *
     * @param absoluteRid
     * @return
     */
    public Input setAbsoluteRid(Integer absoluteRid) {
        this.absoluteRid = absoluteRid;
        return this;
    }

    /**
     * The relative relational id (rid) determines the relative position of this inputs rid with respect to
     * other inputs of this neuron.
     *
     * @param relativeRid
     * @return
     */
    public Input setRelativeRid(Integer relativeRid) {
        this.relativeRid = relativeRid;
        return this;
    }

    /**
     * If set to true then the range of this inputs activation needs to match.
     *
     * @return
     */
    public Input setStartRangeMatch(Operator rm) {
        this.startRangeMatch = rm;
        return this;
    }

    public Input setEndRangeMatch(Operator rm) {
        this.endRangeMatch = rm;
        return this;
    }


    public enum RangeRelation {
        EQUALS,
        CONTAINS,
        CONTAINED_IN,
        NONE
    }


    /**
     * setRangeMatch is just a convenience function to call both setStartRangeMatch and setEndRangeMatch at the same time.
     *
     * @param rr
     * @return
     */
    public Input setRangeMatch(RangeRelation rr) {
        switch(rr) {
            case EQUALS:
                setStartRangeMatch(Operator.EQUALS);
                setEndRangeMatch(Operator.EQUALS);
                break;
            case CONTAINS:
                setStartRangeMatch(Operator.LESS_THAN);
                setEndRangeMatch(Operator.GREATER_THAN);
                break;
            case CONTAINED_IN:
                setStartRangeMatch(Operator.GREATER_THAN);
                setEndRangeMatch(Operator.LESS_THAN);
                break;
            default:
                setStartRangeMatch(Operator.NONE);
                setEndRangeMatch(Operator.NONE);
        }
        return this;
    }


    /**
     * setRangeOutput is just a convenience function to call setStartRangeOutput and setEndRangeOutput at the same time.
     *
     * @param ro
     * @return
     */
    public Input setRangeOutput(boolean ro) {
        setStartRangeOutput(ro);
        setEndRangeOutput(ro);
        return this;
    }


    /**
     * Determines if this input is used to compute the range start of the output activation.
     *
     * @param ro
     * @return
     */
    public Input setStartRangeOutput(boolean ro) {
        this.startRangeOutput = ro;
        return this;
    }


    /**
     * Determines if this input is used to compute the range end of the output activation.
     *
     * @param ro
     * @return
     */
    public Input setEndRangeOutput(boolean ro) {
        this.endRangeOutput = ro;
        return this;
    }


    public Input setStartRangeMapping(Mapping startMapping) {
        this.startMapping = startMapping;
        return this;
    }


    public Input setEndRangeMapping(Mapping endMapping) {
        this.endMapping = endMapping;
        return this;
    }


    @Override
    public int compareTo(Input in) {
        int r = Double.compare(weight, in.weight);
        if(r != 0) return r;
        r = Double.compare(minInput, in.minInput);
        if(r != 0) return r;
        r = Boolean.compare(optional, in.optional);
        if(r != 0) return r;
        r = neuron.compareTo(in.neuron);
        if(r != 0) return r;
        r = startRangeMatch.compareTo(in.startRangeMatch);
        if(r != 0) return r;
        r = endRangeMatch.compareTo(in.endRangeMatch);
        if(r != 0) return r;
        r = Utils.compareInteger(relativeRid, in.relativeRid);
        if (r != 0) return r;
        r = Utils.compareInteger(absoluteRid, in.absoluteRid);
        if (r != 0) return r;
        r = Boolean.compare(startRangeOutput, in.startRangeOutput);
        if (r != 0) return r;
        r = Boolean.compare(endRangeOutput, in.endRangeOutput);
        if (r != 0) return r;
        r = Utils.compareInteger(startMapping.ordinal(), in.startMapping.ordinal());
        if (r != 0) return r;
        r = Utils.compareInteger(endMapping.ordinal(), in.endMapping.ordinal());
        return r;
    }
}
