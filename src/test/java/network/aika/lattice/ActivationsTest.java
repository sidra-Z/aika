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
import network.aika.neuron.INeuron;
import network.aika.neuron.Neuron;
import network.aika.neuron.Synapse;
import network.aika.neuron.activation.Activation;
import network.aika.neuron.activation.Position;
import network.aika.neuron.relation.Relation;
import org.junit.Assert;
import org.junit.Test;

import java.util.Map;
import java.util.TreeMap;

import static network.aika.neuron.INeuron.Type.EXCITATORY;
import static network.aika.neuron.Synapse.OUTPUT;
import static network.aika.neuron.activation.Activation.BEGIN;
import static network.aika.neuron.activation.Activation.END;
import static network.aika.neuron.relation.Relation.EQUALS;

/**
 *
 * @author Lukas Molzberger
 */
public class ActivationsTest {


    @Test
    public void addActivationsTest() {
        Model m = new Model();

        Neuron inA = m.createNeuron("A");

        Neuron pA = Neuron.init(m.createNeuron("pA"), 0.5, EXCITATORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(inA)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS)
        );

        Document doc = m.createDocument("aaaaaaaaaa", 0);

        inA.addInput(doc, 0, 1);
        inA.addInput(doc, 2, 3);

        Assert.assertNotNull(pA.getActivation(doc, 0, 1, false));
        Assert.assertNull(pA.getActivation(doc, 1, 2, false));
        Assert.assertNotNull(pA.getActivation(doc, 2, 3, false));

        inA.addInput(doc, 1, 2);

        Assert.assertNotNull(pA.getActivation(doc, 0, 1, false));
        Assert.assertNotNull(pA.getActivation(doc, 1, 2, false));
        Assert.assertNotNull(pA.getActivation(doc, 2, 3, false));
    }
}
