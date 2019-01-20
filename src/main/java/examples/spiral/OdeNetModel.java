package examples.spiral;

import org.deeplearning4j.nn.graph.ComputationGraph;

class OdeNetModel implements ModelFactory {

    @Override
    public ComputationGraph create(long nrofSamples) {
        return null;
    }

    @Override
    public String name() {
        return "odenet";
    }
}
