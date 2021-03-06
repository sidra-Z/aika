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
import network.aika.neuron.Neuron;
import network.aika.neuron.Synapse;
import network.aika.neuron.INeuron;
import network.aika.neuron.relation.Relation;
import org.junit.Assert;
import org.junit.Test;

import java.util.stream.Collectors;

import static network.aika.neuron.INeuron.Type.EXCITATORY;
import static network.aika.neuron.INeuron.Type.INPUT;
import static network.aika.neuron.Synapse.OUTPUT;
import static network.aika.neuron.relation.Relation.BEGIN_EQUALS;
import static network.aika.neuron.relation.Relation.END_EQUALS;
import static network.aika.neuron.relation.Relation.EQUALS;

/**
 *
 * @author Lukas Molzberger
 */
public class PatternLatticeTest {

    @Test
    public void testPredefinedPatterns() {
        Model m = new Model();
        Neuron inA = m.createNeuron("A", INPUT);
        Neuron inB = m.createNeuron("B", INPUT);
        Neuron inC = m.createNeuron("C", INPUT);

        Neuron.init(m.createNeuron("ABC", EXCITATORY),
                0.001,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(inA)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(1)
                        .setNeuron(inB)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(2)
                        .setNeuron(inC)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(1)
                        .setRelation(EQUALS),
                new Relation.Builder()
                        .setFrom(1)
                        .setTo(2)
                        .setRelation(EQUALS),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS),
                new Relation.Builder()
                        .setFrom(1)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS),
                new Relation.Builder()
                        .setFrom(2)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS)
        );

        Document doc = new Document(m, "", 0);


        InputNode pA = inA.get().getOutputNode().get();
        InputNode pB = inB.get().getOutputNode().get();
        InputNode pC = inC.get().getOutputNode().get();

        AndNode pAB = pA.andChildren.firstEntry().getValue().child.get();
        Assert.assertNotNull(pAB.getProvider());

        AndNode pBC = pC.andChildren.firstEntry().getValue().child.get();
        Assert.assertNotNull(pBC.getProvider());

        Assert.assertEquals(1, pAB.andChildren.size());
        Assert.assertEquals(1, pBC.andChildren.size());

        AndNode pABC = pAB.andChildren.firstEntry().getValue().child.get();
        Assert.assertNotNull(pABC);

        Assert.assertEquals(2, pABC.parents.size());
    }


    @Test
    public void testMultipleActivation() {
        Model m = new Model();

        Neuron inA = m.createNeuron("A", INPUT);
        Neuron inB = m.createNeuron("B", INPUT);
        Neuron inC = m.createNeuron("C", INPUT);

        Neuron nABC = Neuron.init(m.createNeuron("ABC", EXCITATORY),
                0.001,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(inA)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(1)
                        .setNeuron(inB)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(2)
                        .setNeuron(inC)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(1)
                        .setRelation(EQUALS),
                new Relation.Builder()
                        .setFrom(1)
                        .setTo(2)
                        .setRelation(EQUALS),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(OUTPUT)
                        .setRelation(BEGIN_EQUALS),
                new Relation.Builder()
                        .setFrom(2)
                        .setTo(OUTPUT)
                        .setRelation(END_EQUALS)
        );

        Document doc = new Document(m, "aaaaaaaaaa", 0);

        inA.addInput(doc, 0, 1);
        inB.addInput(doc, 0, 1);
        inC.addInput(doc, 0, 1);

        Assert.assertFalse(nABC.getActivations(doc, false).collect(Collectors.toList()).isEmpty());
    }
}
