package examples.spiral;

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;

/**
 * Applies an {@link InputPreProcessor} to the labels of a given {@link MultiDataSetPreProcessor}
 *
 * @author Christian Skarby
 */
public class LabelsPreProcessor implements MultiDataSetPreProcessor {

    private final InputPreProcessor preProcessor;

    public LabelsPreProcessor(InputPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public void preProcess(MultiDataSet multiDataSet) {
        final INDArray newLabels = preProcessor.preProcess(
                multiDataSet.getLabels(0),
                (int) multiDataSet.getLabels(0).size(0),
                LayerWorkspaceMgr.noWorkspaces());
        multiDataSet.setLabels(0, newLabels);
    }
}
