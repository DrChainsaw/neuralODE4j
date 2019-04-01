package examples.spiral;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;

/**
 * Deserializes model from file if the file exists, otherwise a new model is created from a source {@link ModelFactory}
 *
 * @author Christian Skarby
 */
class DeserializingModelFactory implements ModelFactory {

    private static final Logger log = LoggerFactory.getLogger(DeserializingModelFactory.class);

    private final File file;
    private final ModelFactory source;

    DeserializingModelFactory(File file, ModelFactory source) {
        this.file = file;
        this.source = source;
    }

    @Override
    public TimeVae createNew(long nrofSamples, double noiseSigma, long nrofLatentDims) {
        if(file.exists()) {
            try {
                log.info("Restore computation graph from " + file);
                return createFrom(ModelSerializer.restoreComputationGraph(file, true));
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
        log.info("Creating new graph as " + file + " does not exist");
        return source.createNew(nrofSamples, noiseSigma, nrofLatentDims);
    }

    @Override
    public TimeVae createFrom(ComputationGraph graph) {
        return source.createFrom(graph);
    }

    @Override
    public String name() {
        return source.name();
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor(long nrofLatentDims) {
        return source.getPreProcessor(nrofLatentDims);
    }
}
