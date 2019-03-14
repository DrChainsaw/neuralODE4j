package examples.spiral.vertex.conf;

/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.InputTypeUtil;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;

import java.util.Collection;
import java.util.Map;


/**
 * Loss layer which is transparent in the sense that it just forwards its input to a given {@link ILossFunction}
 *
 * @author Christian Skarby
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class LossLayerTransparent extends FeedForwardLayer {

    protected ILossFunction lossFn;

    private LossLayerTransparent(Builder builder) {
        super(builder);
        this.lossFn = builder.lossFn;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        examples.spiral.vertex.impl.LossLayerTransparent ret =
                new examples.spiral.vertex.impl.LossLayerTransparent(conf);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return EmptyParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input type for LossLayerTransparent (layer index = " + layerIndex
                    + ", layer name=\"" + getLayerName() + "\"): Expected RNN input, got " + inputType);
        }
        return inputType;
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType, getLayerName());
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        //During inference and training: dup the input array. But, this counts as *activations* not working memory
        return new LayerMemoryReport.Builder(layerName, LossLayerTransparent.class, inputType, inputType).standardMemory(0, 0) //No params
                .workingMemory(0, 0, 0, 0)
                .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                .build();
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        //No op
    }


    public static class Builder extends FeedForwardLayer.Builder<LossLayerTransparent.Builder> {

        private ILossFunction lossFn;

        public Builder() {

        }

        /**
         * @param lossFunction Loss function for the loss layer
         */
        public Builder lossFunction(ILossFunction lossFunction) {
            this.lossFn = lossFunction;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public LossLayerTransparent build() {
            return new LossLayerTransparent(this);
        }
    }
}

