package examples.cifar10;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
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
    public ComputationGraph create() {
        if(file.exists()) {
            try {
                log.info("Restore computation graph from " + file);
                return ModelSerializer.restoreComputationGraph(file, true);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
        log.info("Creating new graph as " + file + " does not exist");
        return source.create();
    }

    @Override
    public String name() {
        return source.name();
    }

    @Override
    public MultiDataSetIterator wrapIter(DataSetIterator iter) {
        return source.wrapIter(iter);
    }
}
