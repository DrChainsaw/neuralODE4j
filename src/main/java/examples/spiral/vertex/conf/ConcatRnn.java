package examples.spiral.vertex.conf;

import lombok.Data;
import lombok.val;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseRecurrentLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.LayerValidation;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * Very simple RNN which concatenates input and a hidden state instead of adding them.
 * It implements {@code out_t = activationFn( concat(in_t, out_(t-1)) * inWeight + bias)}.
 *
 * Same type of RNN as used in https://github.com/rtqichen/torchdiffeq/blob/master/examples/latent_ode.py
 *
 * @author Christian Skarby
 */
@Data
public class ConcatRnn extends BaseRecurrentLayer {

    private boolean addHiddenToNin = false;

    protected ConcatRnn(ConcatRnn.Builder builder) {
        super(builder);
    }

    private ConcatRnn() {

    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams, DataType networkDataType) {
        LayerValidation.assertNInNOutSet("ConcatRnn", getLayerName(), layerIndex, getNIn(), getNOut());

        examples.spiral.vertex.impl.ConcatRnn ret =
                new examples.spiral.vertex.impl.ConcatRnn(conf, networkDataType);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        setNIn(getNIn() + getNOut());
        Map<String, INDArray> paramTable = DefaultParamInitializer.getInstance().init(conf, layerParamsView, initializeParams);
        setNIn(getNIn() - getNOut());
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return new DefaultParamInitializer() {
            @Override
            public long numParams(org.deeplearning4j.nn.conf.layers.Layer l) {
                FeedForwardLayer layerConf = (FeedForwardLayer) l;
                val nOut = layerConf.getNOut();
                val nIn = layerConf.getNIn() + nOut;
                return (nIn * nOut + (hasBias(l) ? nOut : 0)); //weights + bias
            }
        };
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return null;
    }

    public static class Builder extends BaseRecurrentLayer.Builder<ConcatRnn.Builder> {

        @Override
        public ConcatRnn build() {
            return new ConcatRnn(this);
        }
    }
}
