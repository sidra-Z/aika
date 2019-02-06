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
package network.aika;


import network.aika.lattice.InputNode;
import network.aika.lattice.Node;
import network.aika.neuron.INeuron;
import network.aika.neuron.Neuron;
import network.aika.Provider.SuspensionMode;
import network.aika.neuron.activation.Linker;
import network.aika.neuron.activation.SearchNode;

import java.lang.ref.WeakReference;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;


/**
 * The model consists of two layers. The first layer is the actual neural network consisting of neurons and synapses.
 * The second layer is a pattern lattice containing a boolean logic representation of all the neurons. Whenever the
 * synapse weights of a neuron are adjusted, then the underlying boolean logic representation of this neuron will be
 * updated too.
 * <p>
 * <p>The model supports parallel processing using a fixed number of threads.
 *
 * @author Lukas Molzberger
 */
public class Model {

    public static boolean ENABLE_RELATION_WEIGHTS = false;


    public int numberOfThreads = 1;

    public int[] lastCleanup;

    public Document[] docs;

    public SuspensionHook suspensionHook;

    private WritableFactory nodeExtensionFactory;
    private WritableFactory neuronExtensionFactory;
    private WritableFactory synapseExtensionFactory;
    private WritableFactory activationExtensionFactory;

    public LinkerFactory linkerFactory = (doc) -> new Linker(doc);
    public SearchNode.SkipSelectStep skipSelectStep = (act) -> false;

    public AtomicInteger docIdCounter = new AtomicInteger(0);
    public AtomicInteger currentId = new AtomicInteger(0);

    // Important: the id field needs to be referenced by the provider!
    public WeakHashMap<Integer, WeakReference<Provider<? extends AbstractNode>>> providers = new WeakHashMap<>();
    public Map<Integer, Provider<? extends AbstractNode>> activeProviders = new TreeMap<>();

    public Map<Integer, PassiveInputFunction> passiveActivationFunctions = new TreeMap<>();

    public int defaultThreadId = 0;


    public static AtomicLong visitedCounter = new AtomicLong(1);


    /**
     * Creates a model with a single thread.
     */
    public Model() {
        this(null, 1);
    }


    public Model(SuspensionHook sh, int numberOfThreads) {
        assert numberOfThreads >= 1;
        this.numberOfThreads = numberOfThreads;

        lastCleanup = new int[numberOfThreads];
        docs = new Document[numberOfThreads];
        suspensionHook = sh;
    }


    public SuspensionHook getSuspensionHook() {
        return suspensionHook;
    }


    public void setSuspensionHook(SuspensionHook suspensionHook) {
        this.suspensionHook = suspensionHook;
    }


    public WritableFactory getNodeExtensionFactory() {
        return nodeExtensionFactory;
    }


    public void setNodeExtensionFactory(WritableFactory nodeExtensionFactory) {
        this.nodeExtensionFactory = nodeExtensionFactory;
    }

    public WritableFactory getNeuronExtensionFactory() {
        return neuronExtensionFactory;
    }


    public void setNeuronExtensionFactory(WritableFactory neuronExtensionFactory) {
        this.neuronExtensionFactory = neuronExtensionFactory;
    }

    public WritableFactory getSynapseExtensionFactory() {
        return synapseExtensionFactory;
    }


    public void setSynapseExtensionFactory(WritableFactory synapseExtensionFactory) {
        this.synapseExtensionFactory = synapseExtensionFactory;
    }


    public WritableFactory getActivationExtensionFactory() {
        return activationExtensionFactory;
    }


    public void setActivationExtensionFactory(WritableFactory activationExtensionFactory) {
        this.activationExtensionFactory = activationExtensionFactory;
    }


    public LinkerFactory getLinkerFactory() {
        return linkerFactory;
    }

    public void setLinkerFactory(LinkerFactory linkerFactory) {
        this.linkerFactory = linkerFactory;
    }


    public SearchNode.SkipSelectStep getSkipSelectStep() {
        return skipSelectStep;
    }

    public void setSkipSelectStep(SearchNode.SkipSelectStep skipSelectStep) {
        this.skipSelectStep = skipSelectStep;
    }


    public Neuron createNeuron() {
        return createNeuron(null);
    }


    public Neuron createNeuron(String label) {
        return createNeuron(label, null);
    }


    public Neuron createNeuron(String label, String outputText) {
        INeuron n = new INeuron(this, label, outputText);

        InputNode iNode = InputNode.add(this, n);
        iNode.setModified();

        return n.provider;
    }


    public Document createDocument(String txt) {
        return createDocument(txt, 0);
    }


    public Document createDocument(String txt, int threadId) {
        Document doc = new Document(docIdCounter.addAndGet(1), txt, this, threadId);

        if (txt != null) {
            if (docs[threadId] != null) {
                throw new RuntimeException("Two documents are using the same thread. Call clearActivations() first, before processing the next document.");
            }
            docs[threadId] = doc;
        }

        return doc;
    }


    public <P extends Provider<? extends Node>> P lookupNodeProvider(int id) {
        synchronized (providers) {
            WeakReference<Provider<? extends AbstractNode>> wr = providers.get(id);
            if(wr != null) {
                P p = (P) wr.get();
                if (p != null) {
                    return p;
                }
            }

            return (P) new Provider(this, id);
        }
    }



    public Neuron lookupNeuron(int id) {
        synchronized (providers) {
            WeakReference<Provider<? extends AbstractNode>> wr = providers.get(id);
            if(wr != null) {
                Neuron n = (Neuron) wr.get();
                if (n != null) {
                    return n;
                }
            }

            return new Neuron(this, id);
        }
    }


    public void register(Provider p) {
        synchronized (activeProviders) {
            activeProviders.put(p.id, p);
        }
    }


    public void unregister(Provider p) {
        synchronized (activeProviders) {
            activeProviders.remove(p.id);
        }
    }



    /**
     * Suspend all neurons and logic nodes whose last used document id is lower/older than docId.
     *
     * @param docId
     */
    public void suspendUnusedNodes(int docId, SuspensionMode sm) {
        docId = Math.min(docId, getOldestDocIdInProcessing());
        List<Provider> tmp;
        synchronized (activeProviders) {
            tmp = new ArrayList<>(activeProviders.values());
        }
        for (Provider p: tmp) {
            suspend(docId, p, sm);
        }
    }


    public int getOldestDocIdInProcessing() {
        int oldestDocId = Integer.MAX_VALUE;
        for(Document doc: docs) {
            if(doc != null) oldestDocId = Math.min(oldestDocId, doc.id);
        }
        return oldestDocId;
    }


    /**
     * Suspend all neurons and logic nodes in memory.
     *
     */
    public void suspendAll(SuspensionMode sm) {
        suspendUnusedNodes(Integer.MAX_VALUE, sm);
    }


    private boolean suspend(int docId, Provider<? extends AbstractNode> p, SuspensionMode sm) {
        AbstractNode an = p.getIfNotSuspended();
        if (an != null && an.lastUsedDocumentId < docId) {
            p.suspend(sm);
            return true;
        }
        return false;
    }

    public void removeProvider(Provider p) {
        synchronized (activeProviders) {
            activeProviders.remove(p.id);
        }
        synchronized (providers) {
            providers.remove(p.id);
        }
    }


    public interface WritableFactory {

        Writable createObject();
    }

    public interface LinkerFactory {

        Linker createLinker(Document doc);
    }
}
