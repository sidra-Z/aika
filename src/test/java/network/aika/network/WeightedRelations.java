package network.aika.network;

import network.aika.Document;
import network.aika.Model;
import network.aika.neuron.INeuron;
import network.aika.neuron.Neuron;
import network.aika.neuron.Synapse;
import network.aika.neuron.activation.Activation;
import network.aika.neuron.relation.Relation;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

import static network.aika.neuron.Synapse.OUTPUT;
import static network.aika.neuron.relation.Relation.BEGIN_EQUALS;
import static network.aika.neuron.relation.Relation.END_EQUALS;
import static network.aika.neuron.relation.Relation.END_TO_BEGIN_EQUALS;

public class WeightedRelations {


    @Test
    public void testWightedRelations() {
        Model m = new Model();


        Neuron inA = m.createNeuron("IN-A");
        Neuron inB = m.createNeuron("IN-B");

        Neuron pattern = Neuron.init(
                m.createNeuron("AB"),
                1.0,
                INeuron.Type.EXCITATORY,
                new Synapse.Builder()
                        .setSynapseId(0)
                        .setNeuron(inA)
                        .setWeight(10.0)
                        .setBias(-10.0)
                        .setRecurrent(false),
                new Synapse.Builder()
                        .setSynapseId(1)
                        .setNeuron(inB)
                        .setWeight(10.0)
                        .setBias(-10.0)
                        .setRecurrent(false),
                new Relation.Builder()
                        .setFrom(0)
                        .setTo(1)
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


        Document doc = m.createDocument("AB", 0);

        inA.addInput(doc, 0, 1);
        inB.addInput(doc, 1, 2);

        doc.process();



        System.out.println("Output activation:");
        INeuron n = pattern.get();
        for(Activation act: n.getActivations(doc, false).collect(Collectors.toList())) {
            System.out.println("Text Range: " + act.slotsToString());
            System.out.println("Node: " + act.node);
            System.out.println();
        }

        System.out.println("All activations:");
        System.out.println(doc.activationsToString());
        System.out.println();

        doc.clearActivations();
    }

}
