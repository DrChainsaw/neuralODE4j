package ode.vertex.impl.gradview;

import lombok.Data;
import lombok.EqualsAndHashCode;
import ode.vertex.impl.gradview.parname.Concat;
import ode.vertex.impl.gradview.parname.ParamNameMapping;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * {@link GradientViewFactory} which selects either a {@link Contiguous1DView} or a {@link NonContiguous1DView} based on
 * presence of blacklisted parameters in the graph.
 *
 * @author Christian Skarby
 */
@Data
@EqualsAndHashCode
public class GradientViewSelectionFromBlacklisted implements GradientViewFactory {

    private final List<String> nonGradientParamNames;
    private final ParamNameMapping paramNameMapping;

    public GradientViewSelectionFromBlacklisted() {
        this(Arrays.asList(
                BatchNormalizationParamInitializer.GLOBAL_LOG_STD,
                BatchNormalizationParamInitializer.GLOBAL_VAR,
                BatchNormalizationParamInitializer.GLOBAL_MEAN));
    }

    public GradientViewSelectionFromBlacklisted(List<String> nonGradientParamNames) {
        this(nonGradientParamNames,
                new Concat());
    }

    public GradientViewSelectionFromBlacklisted(@JsonProperty("nonGradientParamNames") List<String> nonGradientParamNames,
                                                @JsonProperty("paramNameMapping") ParamNameMapping paramNameMapping) {
        this.nonGradientParamNames = nonGradientParamNames;
        this.paramNameMapping = paramNameMapping;
    }

    public ParameterGradientView create(ComputationGraph graph) {

        final Gradient gradient = getAllGradients(graph);

        for (GraphVertex vertex : graph.getVertices()) {
            if (hasNonGradient(vertex)) {
                return new ParameterGradientView(gradient, createNonContiguous1DView(graph));
            }
        }

        return new ParameterGradientView(gradient, new Contiguous1DView(graph.getGradientsViewArray()));
    }

    private boolean hasNonGradient(GraphVertex vertex) {
        boolean anyNonGrad = false;
        for (String parName : vertex.paramTable(false).keySet()) {
            anyNonGrad |= nonGradientParamNames.contains(parName);
        }
        return anyNonGrad;
    }

    private NonContiguous1DView createNonContiguous1DView(ComputationGraph graph) {
        final NonContiguous1DView gradView = new NonContiguous1DView();

        for (GraphVertex vertex : graph.getVertices()) {
            addGradientView(gradView, vertex);
        }
        return gradView;
    }

    private void addGradientView(NonContiguous1DView gradView, GraphVertex vertex) {
        if (vertex.numParams() > 0 && hasNonGradient(vertex)) {
            Layer layer = vertex.getLayer();

            if (layer == null) {
                // Only way I have found to get mapping from gradient view to gradient view per parameter is though
                // a ParameterInitializer as done below and only Layers seem be able to provide them
                throw new UnsupportedOperationException("Can not (reliably) get correct gradient views from non-layer " +
                        "vertices with blacklisted parameters!");
            }

            Map<String, INDArray> gradParams = layer.conf().getLayer().initializer().getGradientsFromFlattened(layer.conf(), layer.getGradientsViewArray());
            for (Map.Entry<String, INDArray> parNameAndGradView : gradParams.entrySet()) {
                final String parName = parNameAndGradView.getKey();
                final INDArray grad = parNameAndGradView.getValue();

                if (!nonGradientParamNames.contains(parName)) {
                    gradView.addView(grad);
                }
            }
        } else if (vertex.numParams() > 0) {
            gradView.addView(vertex.getGradientsViewArray());
        }
    }

    private Gradient getAllGradients(ComputationGraph graph) {
        final Gradient allGradients = new DefaultGradient(graph.getGradientsViewArray());
        for (GraphVertex vertex : graph.getVertices()) {
            if (vertex.numParams() > 0) {
                Layer layer = vertex.getLayer();

                if (layer == null) {
                    // Only way I have found to get mapping from gradient view to gradient view per parameter is though
                    // a ParameterInitializer as done below and only Layers seem be able to provide them
                    throw new UnsupportedOperationException("Can not (reliably) get correct gradient views from non-layer " +
                            "vertices with blacklisted parameters!");
                }

                Map<String, INDArray> gradParams = layer.conf().getLayer().initializer().getGradientsFromFlattened(layer.conf(), layer.getGradientsViewArray());
                for (Map.Entry<String, INDArray> parNameAndGradView : gradParams.entrySet()) {
                    final String parName = parNameAndGradView.getKey();
                    final INDArray grad = parNameAndGradView.getValue();
                    allGradients.setGradientFor(paramNameMapping.map(vertex.getVertexName(), parName), grad);
                }
            }
        }
        return allGradients;
    }

    @Override
    public ParamNameMapping paramNameMapping() {
        return paramNameMapping;
    }

    @Override
    public GradientViewFactory clone() {
        return new GradientViewSelectionFromBlacklisted(nonGradientParamNames, paramNameMapping);
    }
}
