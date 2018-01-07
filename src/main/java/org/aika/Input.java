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

import org.aika.corpus.Range;
import org.aika.corpus.Range.Relation;
import org.aika.corpus.Range.Output;
import org.aika.corpus.Range.Operator;
import org.aika.corpus.Range.Mapping;
import org.aika.neuron.Synapse;


/**
 * The {@code Input} class is just a helper class which is used to initialize a neuron. Most of the parameters of this class
 * will be mapped to a input synapse for this neuron.
 *
 * @author Lukas Molzberger
 */
public class Input implements Comparable<Input> {
    Neuron neuron;
    double weight;
    double bias;

    Relation rangeMatch = Relation.NONE;
    Output rangeOutput = Output.NONE;

    Integer relativeRid;
    Integer absoluteRid;


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
    public Input setRecurrent(boolean recurrent) {
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
     * The synapse weight of this input.
     *
     * @param weight
     * @return
     */
    public Input setWeight(double weight) {
        this.weight = weight;
        return this;
    }

    /**
     * The bias of this input that will later on be added to the neurons bias.
     *
     * @param bias
     * @return
     */
    public Input setBias(double bias) {
        this.bias = bias;
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
     * <code>setRangeMatch</code> is just a convenience function to call both <code>setBeginToBeginRangeMatch</code> and <code>setEndToEndRangeMatch</code> at the same time.
     *
     * @param rr
     * @return
     */
    public Input setRangeMatch(Relation rr) {
        rangeMatch = rr;
        return this;
    }


    /**
     * <code>setRangeMatch</code> is just a convenience function to call both <code>setBeginToBeginRangeMatch</code> and <code>setEndToEndRangeMatch</code> at the same time.
     *
     * @param beginToBegin
     * @param endToEnd
     * @return
     */
    public Input setRangeMatch(Operator beginToBegin, Operator endToEnd) {
        rangeMatch = Range.Relation.create(beginToBegin, endToEnd);
        return this;
    }


    /**
     * <code>setRangeOutput</code> is just a convenience function to call <code>setBeginRangeOutput</code> and <code>setEndRangeOutput</code> at the same time.
     *
     * @param ro
     * @return
     */
    public Input setRangeOutput(boolean ro) {
        this.rangeOutput = ro ? Output.DIRECT : Output.NONE;
        return this;
    }

    /**
     * <code>setRangeOutput</code> is just a convenience function to call <code>setBeginRangeOutput</code> and <code>setEndRangeOutput</code> at the same time.
     *
     * @param begin
     * @param end
     * @return
     */
    public Input setRangeOutput(boolean begin, boolean end) {
        return setRangeOutput(begin ? Mapping.BEGIN : Mapping.NONE, end ? Mapping.END : Mapping.NONE);
    }


    /**
     * <code>setRangeOutput</code> is just a convenience function to call <code>setBeginRangeOutput</code> and <code>setEndRangeOutput</code> at the same time.
     *
     * @param rangeOutput
     * @return
     */
    public Input setRangeOutput(Output rangeOutput) {
        this.rangeOutput = rangeOutput;
        return this;
    }


    /**
     * Determines if this input is used to compute the range start of the output activation.
     *
     * @param begin
     * @param end
     * @return
     */
    public Input setRangeOutput(Mapping begin, Mapping end) {
        this.rangeOutput = Output.create(begin, end);
        return this;
    }


    protected Synapse getSynapse(Neuron outputNeuron) {
        Synapse s = new Synapse(
                neuron,
                outputNeuron,
                new Synapse.Key(
                        relativeRid,
                        absoluteRid,
                        rangeMatch,
                        rangeOutput
                )
        );

        Synapse os = outputNeuron.get().inputSynapses.get(s);
        if(os != null) return os;

        os = neuron.get().outputSynapses.get(s);
        if(os != null) return os;

        return s;
    }


    @Override
    public int compareTo(Input in) {
        int r = neuron.compareTo(in.neuron);
        if(r != 0) return r;
        r = rangeMatch.compareTo(in.rangeMatch);
        if(r != 0) return r;
        r = Utils.compareInteger(relativeRid, in.relativeRid);
        if (r != 0) return r;
        r = Utils.compareInteger(absoluteRid, in.absoluteRid);
        if (r != 0) return r;
        r = rangeOutput.compareTo(in.rangeOutput);
        return r;
    }
}
