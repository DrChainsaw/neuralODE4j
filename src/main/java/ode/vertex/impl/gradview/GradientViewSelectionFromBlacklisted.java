package ode.vertex.impl.gradview;

import lombok.Data;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * {@link GradientViewFactory} which selects either a XX or a {@link NonContiguous1DView} based on presence of blacklisted
 * parameters in the graph.
 *
 * @author Christian Skarby
 */
@Data
public class GradientViewSelectionFromBlacklisted implements GradientViewFactory {

    private final List<String> nonGradientParamNames;

    public GradientViewSelectionFromBlacklisted() {
        this(Arrays.asList(
                BatchNormalizationParamInitializer.GLOBAL_VAR,
                BatchNormalizationParamInitializer.GLOBAL_MEAN));
    }

    public GradientViewSelectionFromBlacklisted(@JsonProperty("nonGradientParamNames") List<String> nonGradientParamNames) {
        this.nonGradientParamNames = nonGradientParamNames;
    }

    public INDArray1DView create(ComputationGraph graph) {

        for (GraphVertex vertex : graph.getVertices()) {
            if (hasNonGradient(vertex)) {
                return createNonContiguous1DView(graph);
            }
        }

        return new Contiguous1DView(graph.getGradientsViewArray());
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

    @Override
    public GradientViewFactory clone() {
        return new GradientViewSelectionFromBlacklisted(nonGradientParamNames);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof GradientViewSelectionFromBlacklisted)) return false;
        GradientViewSelectionFromBlacklisted that = (GradientViewSelectionFromBlacklisted) o;
        return nonGradientParamNames.equals(that.nonGradientParamNames);
    }

    @Override
    public int hashCode() {
        return Objects.hash(nonGradientParamNames);
    }

}
