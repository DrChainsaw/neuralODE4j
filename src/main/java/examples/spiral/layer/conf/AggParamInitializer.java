package examples.spiral.layer.conf;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Map;

public class AggParamInitializer implements ParamInitializer {

    private final List<Layer> layers;

    public AggParamInitializer(List<Layer> initializers) {
        this.layers = initializers;
    }

    @Override
    public long numParams(NeuralNetConfiguration conf) {
        return numParams();
    }

    @Override
    public long numParams(Layer layer) {
        return numParams();
    }

    private long numParams() {
        long numPars = 0;
        for(Layer layer: layers) {
            numPars += layer.initializer().numParams(layer);
        }
        return numPars;
    }

    @Override
    public List<String> paramKeys(Layer layer) {
        return null;
    }

    @Override
    public List<String> weightKeys(Layer layer) {
        return null;
    }

    @Override
    public List<String> biasKeys(Layer layer) {
        return null;
    }

    @Override
    public boolean isWeightParam(Layer layer, String key) {
        return false;
    }

    @Override
    public boolean isBiasParam(Layer layer, String key) {
        return false;
    }

    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        return null;
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        return null;
    }
}
