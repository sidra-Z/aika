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
package network.aika.network;


import network.aika.Document;
import network.aika.Model;
import network.aika.neuron.Neuron;
import network.aika.neuron.Synapse;
import network.aika.neuron.relation.PositionRelation;
import network.aika.neuron.relation.Relation;
import org.junit.Assert;
import org.junit.Test;

import static network.aika.neuron.INeuron.Type.EXCITATORY;
import static network.aika.neuron.INeuron.Type.INHIBITORY;
import static network.aika.neuron.Synapse.OUTPUT;
import static network.aika.neuron.activation.Activation.BEGIN;
import static network.aika.neuron.activation.Activation.END;
import static network.aika.neuron.relation.Relation.*;

/**
 *
 * @author Lukas Molzberger
 */
public class NegationTest {


    @Test
    public void testTwoNegativeInputs1() {
        Model m = new Model();
        Neuron inA = m.createNeuron("A");
        Neuron inB = m.createNeuron("B");
        Neuron inC = m.createNeuron("C");

        Neuron abcN = m.createNeuron("ABC");

        Neuron.init(abcN,
                5.5,
                EXCITATORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(inA)
                        .setWeight(10.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(1)
                        .setNeuron(inB)
                        .setWeight(-10.0)
                        .setRecurrent(true),
                new Synapse.Builder()
                        .setSynapseId(2)
                        .setNeuron(inC)
                        .setWeight(-10.0)
                        .setRecurrent(true),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(1)
                        .setRelation(CONTAINS),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(2)
                        .setRelation(CONTAINS),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS)
        );

        Document doc = m.createDocument("aaaaaaaaaaa", 0);

        inA.addInput(doc, 0, 11);

        Assert.assertNotNull(abcN.getActivation(doc, 0, 11, false));

        inB.addInput(doc, 2, 7);
        inC.addInput(doc, 4, 9);

        doc.process();

        System.out.println(doc.activationsToString());

        Assert.assertNotNull(abcN.getActivation(doc, 0, 11, false));
    }


    @Test
    public void testTwoNegativeInputs2() {
        Model m = new Model();

        Neuron inA = m.createNeuron("A");

        Neuron inB = m.createNeuron("B");

        Neuron inC = m.createNeuron("C");

        Neuron abcN = m.createNeuron("ABC");

        Neuron outN = Neuron.init(m.createNeuron("OUT"),
                1.0,
                EXCITATORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(abcN)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS)
        );

        Neuron.init(abcN,
                0.001,
                EXCITATORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(inA)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(1)
                        .setNeuron(inB)
                        .setWeight(-1.0)
                        .setRecurrent(true),
                new Synapse.Builder()
                        .setSynapseId(2)
                        .setNeuron(inC)
                        .setWeight(-1.0)
                        .setRecurrent(true),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(1)
                        .setRelation(ANY),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(2)
                        .setRelation(ANY),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS)
        );

        Document doc = m.createDocument("aaaaaaaaaaa", 0);

        inA.addInput(doc, 0, 11);
        inB.addInput(doc, 2, 7);
        inC.addInput(doc, 4, 9);

        System.out.println(doc.activationsToString());

//        Assert.assertNull(Activation.get(t, outN.node, 0, new Range(0, 11), Range.Relation.EQUALS, null, null, null));
    }


    @Test
    public void testSimpleNegation1() {
        Model m = new Model();

        Neuron inA = m.createNeuron("A");

        Neuron asN = m.createNeuron("AS");

        Neuron inS = m.createNeuron("S");

        Neuron outN = Neuron.init(m.createNeuron("OUT"),
                0.0,
                INHIBITORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(asN)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS)
        );

        Neuron.init(asN,
                0.001,
                EXCITATORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(inA)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(1)
                        .setNeuron(inS)
                        .setWeight(-1.0)
                        .setRecurrent(true),
                new Relation.Builder()
                        .setFrom(1)
                        .setTo(0)
                        .setRelation(EQUALS),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS)
        );

        Document doc = m.createDocument("aaaaaaaaaaa", 0);

        inS.addInput(doc, 3, 8);

        System.out.println(doc.activationsToString());

        inA.addInput(doc, 0, 11);

        System.out.println(doc.activationsToString());

        Assert.assertNotNull(outN.getActivation(doc, 0, 11, false));

        doc.clearActivations();
    }


    @Test
    public void testSimpleNegation2() {
        Model m = new Model();

        Neuron inA = m.createNeuron("A");

        Neuron asN = m.createNeuron("AS");

        Neuron inS = m.createNeuron("S");

        Neuron outN = Neuron.init(m.createNeuron("OUT"),
                0.0,
                INHIBITORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(asN)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS)
        );

        Neuron.init(asN,
                0.001,
                EXCITATORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(inA)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(1)
                        .setNeuron(inS)
                        .setWeight(-1.0)
                        .setRecurrent(true),
                new Relation.Builder()
                        .setFrom(1)
                        .setTo(0)
                        .setRelation(CONTAINS),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS)
        );

        Document doc = m.createDocument("aaaaaaaaaaa", 0);

        inS.addInput(doc, 3, 8);


        inA.addInput(doc, 0, 11);

        System.out.println(doc.activationsToString());

        Assert.assertNotNull(outN.getActivation(doc, 0, 11, false));

        doc.clearActivations();
    }


    @Test
    public void testSimpleNegation3() {
        Model m = new Model();

        Neuron inA = m.createNeuron("A");

        Neuron asN = m.createNeuron("AS");

        Neuron inS = m.createNeuron("S");

        Neuron outN = Neuron.init(m.createNeuron("OUT"),
                0.0,
                INHIBITORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(asN)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS)
        );

        Neuron.init(asN,
                0.001,
                EXCITATORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(inA)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(1)
                        .setNeuron(inS)
                        .setWeight(-1.0)
                        .setRecurrent(true),
                new Relation.Builder()
                        .setFrom(1)
                        .setTo(0)
                        .setRelation(CONTAINS),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS)
        );

        Document doc = m.createDocument("aaaaaaaaaaa", 0);

        inA.addInput(doc, 0, 11);
        inS.addInput(doc, 3, 8);

        System.out.println(doc.activationsToString());

        Assert.assertNotNull(outN.getActivation(doc, 0, 11, false));

        doc.clearActivations();
    }


    @Test
    public void testNegation1() {
        Model m = new Model();
        Neuron inA = m.createNeuron("A");
        Neuron inB = m.createNeuron("B");

        Neuron asN = m.createNeuron("AS");
        Neuron absN = m.createNeuron("ABS");
        Neuron bsN = m.createNeuron("BS");

        Neuron inS = Neuron.init(m.createNeuron("S"),
                0.0,
                INHIBITORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(asN)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(1)
                        .setNeuron(absN)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS),
                new Relation.Builder()
                        .setFrom(1)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS)
        );

        Neuron.init(asN,
                0.001,
                EXCITATORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(inA)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(1)
                        .setNeuron(inS)
                        .setWeight(-1.0)
                        .setRecurrent(true),
                new Relation.Builder()
                        .setFrom(1)
                        .setTo(0)
                        .setRelation(CONTAINS),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS)
        );
        Neuron.init(absN,
                0.001,
                EXCITATORY,
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
                        .setNeuron(inS)
                        .setWeight(-1.0)
                        .setRecurrent(true),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(1)
                        .setRelation(new PositionRelation.GreaterThan(END, BEGIN, false)),
                new Relation.Builder()
                        .setFrom(2)
                        .setTo(0)
                        .setRelation(OVERLAPS),
                new Relation.Builder()
                        .setFrom(2)
                        .setTo(1)
                        .setRelation(OVERLAPS),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(OUTPUT)
                        .setRelation(BEGIN_EQUALS),
                new Relation.Builder()
                        .setFrom(1)
                        .setTo(OUTPUT)
                        .setRelation(END_EQUALS)
        );

        {
            Document doc = m.createDocument("aaaaaaaaaa", 0);

            inA.addInput(doc, 0, 6);
            inB.addInput(doc, 0, 6);

            System.out.println(doc.activationsToString());

            Assert.assertNotNull(inS.getActivation(doc, 0, 6, false));
            Assert.assertEquals(2, inS.getActivation(doc, 0, 6, false).getInputLinks(false).count());

            doc.clearActivations();
        }

        {
            Document doc = m.createDocument("aaaaaaaaaa", 0);

            inA.addInput(doc, 0, 6);
            inB.addInput(doc, 3, 9);

            System.out.println(doc.activationsToString());

            Assert.assertNotNull(inS.getActivation(doc, 0, 9, false));
            Assert.assertEquals(1, inS.getActivation(doc, 0, 6, false).getInputLinks(false).count());
            Assert.assertEquals(1, inS.getActivation(doc, 0, 9, false).getInputLinks(false).count());

            doc.clearActivations();
        }
    }


    @Test
    public void testNegation2() {
        Model m = new Model();

        Neuron inA = m.createNeuron("A");
        Neuron inB = m.createNeuron("B");
        Neuron inC = m.createNeuron("C");

        Neuron asN = m.createNeuron("AS");
        Neuron ascN = m.createNeuron("ASC");
        Neuron bsN = m.createNeuron("BS");

        Neuron inS = Neuron.init(m.createNeuron("S"),
                0.0,
                INHIBITORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(asN)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(1)
                        .setNeuron(ascN)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(2)
                        .setNeuron(bsN)
                        .setWeight(1.0)
                        .setRecurrent(false),
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

        Neuron.init(asN,
                0.001,
                EXCITATORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(inA)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(1)
                        .setNeuron(inS)
                        .setWeight(-1.0)
                        .setRecurrent(true),
                new Relation.Builder()
                        .setFrom(1)
                        .setTo(0)
                        .setRelation(EQUALS),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS),
                new Relation.Builder()
                        .setFrom(1)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS)
        );
        Neuron.init(ascN,
                0.001,
                EXCITATORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(inA)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(1)
                        .setNeuron(inC)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(2)
                        .setNeuron(inS)
                        .setWeight(-1.0)
                        .setRecurrent(true),
                new Relation.Builder()
                        .setFrom(1)
                        .setTo(0)
                        .setRelation(EQUALS),
                new Relation.Builder()
                        .setFrom(2)
                        .setTo(0)
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

        Neuron.init(bsN,
                0.001,
                EXCITATORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(inB)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(1)
                        .setNeuron(inS)
                        .setWeight(-1.0)
                        .setRecurrent(true),
                new Relation.Builder()
                        .setFrom(1)
                        .setTo(0)
                        .setRelation(EQUALS),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS),
                new Relation.Builder()
                        .setFrom(1)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS)
        );

        Neuron outA = Neuron.init(m.createNeuron("OUT A"),
                0.0,
                EXCITATORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(asN)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS)
        );
        Neuron outAC = Neuron.init(m.createNeuron("OUT AC"),
                0.0,
                EXCITATORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(ascN)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS)
        );
        Neuron outB = Neuron.init(m.createNeuron("OUT B"),
                1.0,
                EXCITATORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(bsN)
                        .setWeight(1.0)
                        .setRecurrent(false),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS)
        );

        Document doc = m.createDocument("aaaaaaaaaa", 0);


        inA.addInput(doc, 0, 6);
        inB.addInput(doc, 0, 6);
        inC.addInput(doc, 0, 6);

        doc.process();

        System.out.println(doc.activationsToString());
    }




    /**
     *
     *       -----
     *  A ---| &  |------
     *     -*| C  |     |       ------
     *     | ------     |   G---| &  |
     *      \           |       | H  |-----
     *       \/-----------------|    |
     *       /\-----------------|    |
     *      /           |       ------
     *     | ------     |
     *     -*| &  |------
     *  B ---| D  |
     *       ------
     *
     */

    @Test
    public void testOptions() {
        Model m = new Model();

        Neuron inA = m.createNeuron("A");
        Neuron inB = m.createNeuron("B");

        Neuron pC = m.createNeuron("C");
        Neuron pD = m.createNeuron("D");

        Neuron.init(pC,
                0.5,
                EXCITATORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(inA)
                        .setWeight(2.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(1)
                        .setNeuron(pD)
                        .setWeight(-2.0)
                        .setRecurrent(true),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS)
        );

        Neuron.init(pD,
                0.5,
                EXCITATORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(inB)
                        .setWeight(2.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(1)
                        .setNeuron(pC)
                        .setWeight(-2.0)
                        .setRecurrent(true),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(OUTPUT)
                        .setRelation(EQUALS)
        );


        Neuron inG = m.createNeuron("G");

        Neuron pH = Neuron.init(m.createNeuron("H"),
                0.001,
                EXCITATORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(pC)
                        .setWeight(2.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(1)
                        .setNeuron(pD)
                        .setWeight(2.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(2)
                        .setNeuron(inG)
                        .setWeight(2.0)
                        .setRecurrent(false),
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

        Document doc = m.createDocument("aaaaaaaaaa", 0);

        inA.addInput(doc, 0, 1);
        inB.addInput(doc, 0, 1);
        inG.addInput(doc, 0, 1);

        System.out.println(doc.activationsToString());

        Assert.assertNotNull(pC.getActivation(doc, 0, 1, false));
        Assert.assertNotNull(pD.getActivation(doc, 0, 1, false));

        // Die Optionen 0 und 2 stehen in Konflikt. Da sie aber jetzt in Oder Optionen eingebettet sind, werden sie nicht mehr ausgefiltert.
//        Assert.assertNull(pH.node.getFirstActivation(t));
    }


}
