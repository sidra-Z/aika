package network.aika.network;


import network.aika.Document;
import network.aika.Model;
import network.aika.neuron.Neuron;
import network.aika.neuron.Synapse;
import network.aika.neuron.INeuron;
import network.aika.neuron.activation.Activation;
import network.aika.neuron.relation.MultiRelation;
import network.aika.neuron.relation.PositionRelation;
import network.aika.neuron.relation.Relation;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static network.aika.neuron.Synapse.OUTPUT;
import static network.aika.neuron.activation.Activation.BEGIN;
import static network.aika.neuron.activation.Activation.END;
import static network.aika.neuron.relation.Relation.*;
import static network.aika.neuron.relation.PositionRelation.Equals;
import static network.aika.neuron.relation.PositionRelation.LessThan;
import static network.aika.neuron.relation.PositionRelation.GreaterThan;
import static network.aika.neuron.relation.Relation.EQUALS;


public class PositionRelationTest {


    @Test
    public void testRangeRelation() {
        Model m = new Model();

        Neuron inA = m.createNeuron("A");
        Neuron inB = m.createNeuron("B");


        Neuron outC = Neuron.init(m.createNeuron("C"),
                5.0,
                INeuron.Type.EXCITATORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(inA)
                        .setWeight(10.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(1)
                        .setNeuron(inB)
                        .setWeight(10.0)
                        .setRecurrent(false),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(1)
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


        Document doc = m.createDocument("aaaaaaaaaa", 0);

        inA.addInput(doc, 0, 1);
        inB.addInput(doc, 0, 1);

        Activation outC1 = outC.getActivation(doc, 0, 1, false);

        System.out.println(doc.activationsToString());

        Assert.assertNotNull(outC1);
    }

    @Test
    public void testABCPattern() {
        Model m = new Model();

        Neuron inA = m.createNeuron("A");
        Neuron inB = m.createNeuron("B");
        Neuron inC = m.createNeuron("C");

        Neuron outD = Neuron.init(m.createNeuron("D"),
                0.001,
                INeuron.Type.EXCITATORY,
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
                        .setRelation(END_TO_BEGIN_EQUALS),
                new Relation.Builder()
                        .setFrom(1)
                        .setTo(2)
                        .setRelation(END_TO_BEGIN_EQUALS),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(OUTPUT)
                        .setRelation(BEGIN_EQUALS),
                new Relation.Builder()
                        .setFrom(2)
                        .setTo(OUTPUT)
                        .setRelation(END_EQUALS)
        );

        Document doc = m.createDocument("aaaaaaaaaa", 0);

        inA.addInput(doc, 0, 1);
        inB.addInput(doc, 1, 2);
        inC.addInput(doc, 2, 3);

        Activation outD1 = outD.getActivation(doc, 0, 3, false);

        Assert.assertNotNull(outD1);
    }


    @Test
    public void testHuettenheim() {
        Model m = new Model();

        HashMap<Character, Neuron> chars = new HashMap<>();
        for (char c = 'a'; c <= 'z'; c++) {
            Neuron rec = m.createNeuron("IN-" + c);
            chars.put(c, rec);
        }

        String word = "Huettenheim";


        List<Neuron.Builder> inputs = new ArrayList<>();
        for (int i = 0; i < word.length(); i++) {
            char c = word.toLowerCase().charAt(i);

            Neuron rec = chars.get(c);
            if (rec != null) {
                boolean begin = i == 0;
                boolean end = i + 1 == word.length();

                inputs.add(new Synapse.Builder()
                        .setSynapseId(i)
                        .setNeuron(rec)
                        .setWeight(begin || end ? 2.0 : 1.0)
                        .setRecurrent(false)
                );
                if(begin) {
                    inputs.add(
                            new Relation.Builder()
                                    .setFrom(0)
                                    .setTo(OUTPUT)
                                    .setRelation(BEGIN_EQUALS)
                    );
                }

                if(end) {
                    inputs.add(
                            new Relation.Builder()
                                    .setFrom(i)
                                    .setTo(OUTPUT)
                                    .setRelation(END_EQUALS)
                    );
                }

                if(!end) {
                    inputs.add(new Relation.Builder()
                                    .setFrom(i)
                                    .setTo(i + 1)
                                    .setRelation(END_TO_BEGIN_EQUALS)
                    );
                } else {
                    inputs.add(new Relation.Builder()
                                    .setFrom(i)
                                    .setTo(0)
                                    .setRelation(new PositionRelation.GreaterThan(END, BEGIN, false))
                    );
                }
            }
        }

        Neuron n = Neuron.init(m.createNeuron("PATTERN"), 0.5, INeuron.Type.EXCITATORY, inputs.toArray(new Neuron.Builder[inputs.size()]));

        Document doc = m.createDocument("abc Huettenheim cba", 0);

        for (int i = 0; i < doc.length(); i++) {
            char c = doc.getContent().toLowerCase().charAt(i);

            Neuron inputNeuron = chars.get(c);
            if (inputNeuron != null) {
                inputNeuron.addInput(doc, i, i + 1);
            }
        }

        System.out.println(doc.activationsToString());

        assert n.get().getActivations(doc, false).collect(Collectors.toList()).size() >= 1;
    }


    @Test
    public void testOptionalRelation() {
        Model m = new Model();

        Neuron inputChar = m.createNeuron("CHAR");

        Neuron wordPlaceholder = m.createNeuron("Placeholder");


        Neuron pattern = Neuron.init(
                m.createNeuron("BCDEFG"),
                1.0,
                INeuron.Type.EXCITATORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(inputChar)
                        .setWeight(15.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(1)
                        .setNeuron(inputChar)
                        .setWeight(10.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(2)
                        .setNeuron(inputChar)
                        .setWeight(15.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(3)
                        .setNeuron(wordPlaceholder)
                        .setWeight(30.0)
                        .setRecurrent(false),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(3)
                        .setRelation(BEGIN_EQUALS),
                new Relation.Builder()
                        .setFrom(2)
                        .setTo(3)
                        .setRelation(END_EQUALS),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(1)
                        .setRelation(new MultiRelation(
                                new Equals(END, BEGIN, true, true),
                                new LessThan(BEGIN, BEGIN, false, false, false, Integer.MAX_VALUE)
                        )),
                new Relation.Builder()
                        .setFrom(1)
                        .setTo(1)
                        .setRelation(new Equals(END, BEGIN, true, true)),
                new Relation.Builder()
                        .setFrom(1)
                        .setTo(2)
                        .setRelation(new MultiRelation(
                                new Equals(END, BEGIN, true, true),
                                new LessThan(END, END, false, false, false, Integer.MAX_VALUE)
                        )),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(OUTPUT)
                        .setRelation(BEGIN_EQUALS),
                new Relation.Builder()
                        .setFrom(2)
                        .setTo(OUTPUT)
                        .setRelation(END_EQUALS)
        );


        Document doc = m.createDocument("a b c d e f g h ", 0);

        wordPlaceholder.addInput(doc, 2, 14);

        for(int i = 0; i < doc.length(); i++) {
            char c = doc.getContent().charAt(i);
            if(c != ' ') {
                inputChar.addInput(doc, i, i + 2);
            }
        }

        doc.process();

        Assert.assertEquals(1, pattern.get().size(doc));


        System.out.println("Output activation:");
        INeuron n = pattern.get();
        for(Activation act: n.getActivations(doc, false).collect(Collectors.toList())) {
            System.out.println("Text Range: " + act.slotsToString());
            System.out.println("Neuron: " + act.getLabel());
            System.out.println();
        }

        System.out.println("All activations:");
        System.out.println(doc.activationsToString());
        System.out.println();

        doc.clearActivations();
    }

}
