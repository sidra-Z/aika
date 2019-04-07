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
import network.aika.neuron.activation.Activation;

import java.util.*;


public abstract class NodeActivation<T extends Node> implements Comparable<NodeActivation<T>> {

    public final int id;

    private final T node;

    protected final Document doc;

    Long repropagateV;
    public boolean registered;

    TreeMap<Integer, AndNode.Link> outputsToAndNode = new TreeMap<>();
    TreeMap<Integer, Node.Link> outputsToNeurons = new TreeMap<>();


    public NodeActivation(Document doc, T node) {
        this.id = doc.getNewNodeActivationId();
        this.doc = doc;
        this.node = node;
    }


    public T getNode() {
        return node;
    }


    public Document getDocument() {
        return doc;
    }


    public int getThreadId() {
        return doc.getThreadId();
    }

    public abstract Activation getInputActivation(int i);


    @Override
    public int compareTo(NodeActivation<T> act) {
        return Integer.compare(id, act.id);
    }
}
