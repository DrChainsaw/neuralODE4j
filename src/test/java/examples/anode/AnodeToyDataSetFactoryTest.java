package examples.anode;

import com.beust.jcommander.JCommander;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.GreaterThan;
import org.nd4j.linalg.indexing.conditions.LessThan;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.*;

/**
 * Test cases for {@link AnodeToyDataSetFactory}
 *
 * @author Christian Skarby
 */
public class AnodeToyDataSetFactoryTest {

    /**
     * Test that 1D separable data set is actually separable
     */
    @Test
    public void create1DSeparable() {
        final Pair<INDArray, INDArray> data = splitData(createDataSet("-separable"));
        assertTrue("Data is not separable!!",
                data.getFirst().maxNumber().doubleValue() < data.getSecond().minNumber().doubleValue());
    }

    /**
     * Test that 2D separable data set is actually separable
     */
    @Test
    public void create2DSeparable() {
        final Pair<INDArray, INDArray> data = splitData(createDataSet("-separable", "-2D"));
        // Bad testcase: This only tests the average which basically says nothing about separability.
        assertTrue("Data is not separable!!",
                data.getFirst().getColumn(1).norm2Number().doubleValue() > data.getSecond().getColumn(1).norm2Number().doubleValue());
    }

    /**
     * Test that 1D non-separable data set is actually non-separable
     */
    @Test
    public void create1DNonSeparable() {
        final Pair<INDArray, INDArray> data = splitData(createDataSet());
        assertFalse("Data is separable!!",
                data.getFirst().maxNumber().doubleValue() < data.getSecond().maxNumber().doubleValue());
        assertFalse("Data is separable!!",
                data.getFirst().minNumber().doubleValue() > data.getSecond().minNumber().doubleValue());
    }

    /**
     * Test that 2D non-separable data set is actually non-separable
     */
    @Test
    public void create2DNonSeparable() {
        final Pair<INDArray, INDArray> data = splitData(createDataSet("-2D"));
        for(int i = 0; i < data.getFirst().size(1); i++) {
            assertFalse("Data is separable!!",
                    data.getFirst().getColumn(i).maxNumber().doubleValue() < data.getSecond().getColumn(i).maxNumber().doubleValue());
            assertFalse("Data is separable!!",
                    data.getFirst().getColumn(i).minNumber().doubleValue() > data.getSecond().getColumn(i).minNumber().doubleValue());
        }
    }

    private static DataSet createDataSet(String... args) {
        final List<String> allArgs = new ArrayList<>(Arrays.asList(args));
        allArgs.add("-trainBatchSize");
        allArgs.add("1000");
        allArgs.add("-nrofExamples");
        allArgs.add("1000");

        final AnodeToyDataSetFactory factory = new AnodeToyDataSetFactory();
        JCommander.newBuilder()
                .addObject(factory)
                .build().parse(allArgs.toArray(new String[0]));

        final DataSet ds = factory.create().getTrain().next();
        assertEquals("Incorrect number of labels!", 1000, ds.getLabels().size(0));
        assertEquals("Incorrect number of features!", 1000, ds.getFeatures().size(0));

        assertNotEquals("Incorrect label found!", 0d, ds.getLabels().aminNumber().doubleValue(), 1e-10);
        assertEquals("Incorrect max label!", 1d, ds.getLabels().maxNumber().doubleValue(), 1e-10);
        assertEquals("Incorrect min label!", -1d, ds.getLabels().minNumber().doubleValue(), 1e-10);
        return ds;
    }

    private static Pair<INDArray, INDArray> splitData(DataSet ds) {
        final INDArray indsGreater = Nd4j.where(ds.getLabels().cond(new GreaterThan(0)), null, null)[0];
        final INDArray indsLess = Nd4j.where(ds.getLabels().cond(new LessThan(0)), null, null)[0];
        return new Pair<>(
                ds.getFeatures().get(indsLess).reshape(indsLess.length(), ds.getFeatures().size(1)),
                ds.getFeatures().get(indsGreater).reshape(indsGreater.length(), ds.getFeatures().size(1)));
    }
}