package examples.anode;

import com.beust.jcommander.IValueValidator;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIteratorFactory;

public class AnodeToyDataSetFactory implements DataSetIteratorFactory {

    @Parameter(names = "-trainBatchSize", description = "Batch size to use for training")
    private int batchSize = 64;

    @Parameter(names = "-nrofDims", description = "Number of dimensions to use", validateValueWith = OneOrTwo.class)
    private int nrofDims = 1;

    @Parameter(names = "-separable", description = "A separable data set is created if present")
    private boolean separable = false;


    private static class OneOrTwo implements IValueValidator<Integer> {
        @Override
        public void validate(String name, Integer value) throws ParameterException {
            if(value != 1 && value != 2) {
                throw new ParameterException(name + " must be 1 or 2!");
            }
        }
    }

    @Override
    public DataSetIterator create() {
        return null;
    }
}
