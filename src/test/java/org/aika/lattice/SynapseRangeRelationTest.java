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
package org.aika.lattice;


import org.aika.Activation;
import org.aika.Activation.Key;
import org.aika.NeuronActivation;
import org.aika.NeuronActivation.SynapseActivation;
import org.aika.Model;
import org.aika.corpus.Document;
import org.aika.corpus.Range;
import org.aika.corpus.Range.Operator;
import org.aika.corpus.Range.Mapping;
import org.aika.neuron.Neuron;
import org.aika.neuron.Synapse;
import org.junit.Assert;
import org.junit.Test;

import java.util.Collections;

import static org.aika.NeuronActivation.SynapseActivation.INPUT_COMP;

/**
 *
 * @author Lukas Molzberger
 */
public class SynapseRangeRelationTest {


    @Test
    public void testSynapseRangeRelation() {
        Model m = new Model();
        Document doc = m.createDocument("                        ", 0);

        Neuron in = new Neuron(m);
        Neuron on = new Neuron(m);

        Synapse s = new Synapse(in,
                new Synapse.Key(
                        false,
                        false,
                        null,
                        null,
                        Operator.LESS_THAN,
                        Mapping.START,
                        true,
                        Operator.GREATER_THAN,
                        Mapping.END,
                        true
                )
        );
        s.output = on.provider;
        s.link(doc.threadId);

        NeuronActivation iAct0 = in.node.get().addActivationInternal(doc, new Key(in.node.get(), new Range(1, 4), null, doc.bottom), Collections.emptyList(), false);
        NeuronActivation iAct1 = in.node.get().addActivationInternal(doc, new Key(in.node.get(), new Range(6, 7), null, doc.bottom), Collections.emptyList(), false);
        NeuronActivation iAct2 = in.node.get().addActivationInternal(doc, new Key(in.node.get(), new Range(10, 18), null, doc.bottom), Collections.emptyList(), false);
        NeuronActivation oAct = on.node.get().addActivationInternal(doc, new Key(on.node.get(), new Range(6, 7), null, doc.bottom), Collections.emptyList(), false);

        on.linkNeuronRelations(doc, oAct);

        boolean f = false;
        for(SynapseActivation sa: oAct.neuronInputs) {
            if(INPUT_COMP.compare(sa, new SynapseActivation(s, iAct1, oAct)) == 0) f = true;
        }

        Assert.assertTrue(f);
    }

}
