package ode.vertex.impl.helper;

import ode.vertex.conf.helper.GraphInputOutputFactory;
import ode.vertex.impl.gradview.GradientViewFactory;
import ode.vertex.impl.gradview.INDArray1DView;
import ode.vertex.impl.gradview.ParameterGradientView;
import ode.vertex.impl.helper.backward.OdeHelperBackward;
import ode.vertex.impl.helper.forward.OdeHelperForward;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * Helper which jumps through the hoops so that a {@link ComputationGraph} can be seen as the function which provides
 * the derivatives for an ODE solver.
 *
 * @author Christian Skarby
 */
public class OdeGraphHelper {

    private static final Logger log = LoggerFactory.getLogger(OdeGraphHelper.class);

    private final OdeHelperForward odeHelperForward;
    private final OdeHelperBackward odeHelperBackward;
    private final GraphInputOutputFactory graphInputOutputFactory;
    private final CompGraphAsOdeFunction odeFunction;

    public OdeGraphHelper(OdeHelperForward odeHelperForward, OdeHelperBackward odeHelperBackward, GraphInputOutputFactory graphInputOutputFactory, CompGraphAsOdeFunction odeFunction) {
        this.odeHelperForward = odeHelperForward;
        this.odeHelperBackward = odeHelperBackward;
        this.graphInputOutputFactory = graphInputOutputFactory;
        this.odeFunction = odeFunction;
    }

    public static class CompGraphAsOdeFunction {

        private INDArray lastOutput; // z(t1) from paper
        private ParameterGradientView parameterGradientView;
        private final ComputationGraph function;
        private final GradientViewFactory gradientViewFactory;

        public CompGraphAsOdeFunction(ComputationGraph odeFunction, GradientViewFactory gradientViewFactory) {
            this.function = odeFunction;
            this.gradientViewFactory = gradientViewFactory;
        }

        private INDArray lastOutput() {
            return lastOutput;
        }

        private INDArray1DView realGradients() {
            return parameterGradientView.realGradientView();
        }

        private void setLastOutput(INDArray lastOutput) {
            this.lastOutput = lastOutput;
        }

        void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
            function.setBackpropGradientsViewArray(backpropGradientsViewArray);
            parameterGradientView = gradientViewFactory.create(function);
        }

        Map<String, INDArray> paramTable(boolean backpropOnly) {
            final Map<String, INDArray> output = new HashMap<>();
            for (GraphVertex vertex : function.getVertices()) {
                final Map<String, INDArray> partable = vertex.paramTable(backpropOnly);
                if (partable != null) {
                    for (Map.Entry<String, INDArray> parEntry : partable.entrySet()) {
                        output.put(
                                gradientViewFactory.paramNameMapping().map(vertex.getVertexName(), parEntry.getKey()),
                                parEntry.getValue());
                    }
                }
            }
            return output;
        }
    }

    /**
     * Clears the current state wrt training. Gradient views are not touched.
     */
    public void clear() {
        getFunction().clearLayersStates();
        odeFunction.setLastOutput(null);
    }

    public Map<String, INDArray> paramTable(boolean backpropOnly) {
        return odeFunction.paramTable(backpropOnly);
    }

    public ComputationGraph getFunction() {
        return odeFunction.function;
    }

    /**
     * What is this about? Some layers "abuse" the gradient to perform updates of parameters for which no gradient
     * is calculated and this screws up the ODE solvers idea of what the solution is. The following layers are known
     * to do this:
     * <br><br>
     * * BatchNormalization: The global variance and mean are just the (sliding) average of the batch dittos.
     * However, in order to support distributed training the updates are performed by adding
     * the change to the state as a gradient even through it is not really.
     *
     * @param backpropGradientsViewArray View of parameter gradients
     */
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        odeFunction.setBackpropGradientsViewArray(backpropGradientsViewArray);
    }


    public INDArray doForward(LayerWorkspaceMgr workspaceMgr, INDArray[] inputs) {

        final LayerWorkspaceMgr innerWorkspaceMgr = createWorkspaceMgr(workspaceMgr, getFunction());

        getFunction().getConfiguration().setIterationCount(0);
        final INDArray output = odeHelperForward.solve(getFunction(), innerWorkspaceMgr, graphInputOutputFactory.create(inputs));
        log.debug("Nrof func eval forward " + getFunction().getIterationCount());

        odeFunction.setLastOutput(output.detach());

        return output;
    }

    public Pair<Gradient, INDArray[]> doBackward(
            OdeHelperBackward.MiscPar miscPars,
            INDArray lossGradient,
            INDArray[] lastInputs) {

        final OdeHelperBackward.InputArrays inputArrays = new OdeHelperBackward.InputArrays(
                graphInputOutputFactory.create(lastInputs),
                odeFunction.lastOutput(),
                lossGradient,
                odeFunction.realGradients()
        );

        final OdeHelperBackward.MiscPar miscParNewWsMgr = new OdeHelperBackward.MiscPar(
                miscPars.isUseTruncatedBackPropTroughTime(),
                createWorkspaceMgr(miscPars.getWsMgr(), getFunction())
        );

        getFunction().getConfiguration().setIterationCount(0);
        final INDArray[] gradients = odeHelperBackward.solve(getFunction(), inputArrays, miscParNewWsMgr);
        log.debug("Nrof func eval backward " + getFunction().getIterationCount());

        return new Pair<>(odeFunction.parameterGradientView.allGradientsPerParam(), gradients);
    }

    /**
     * Changes names of  workspaces associated with certain {@link ArrayType}s in order to avoid workspace conflicts
     * due to "graph in graph".
     *
     * @param outerWsMgr workspace manager
     * @return LayerWorkspaceMgr with new workspace names but using the same workspace configs as in {@link ComputationGraph}
     */
    private LayerWorkspaceMgr createWorkspaceMgr(final LayerWorkspaceMgr outerWsMgr, ComputationGraph graph) {
        if (outerWsMgr == LayerWorkspaceMgr.noWorkspacesImmutable()) {
            // This can be handled better, but I just CBA to check presence for every array type right now...
            return outerWsMgr;
        }

        return new ComputationGraph(graph.getConfiguration()) {
            LayerWorkspaceMgr spyWsConfigs() {
                // A little bit too many methods to comfortably decorate. Try to copy config instead
                final LayerWorkspaceMgr.Builder wsBuilder = LayerWorkspaceMgr.builder();
                for (ArrayType type : ArrayType.values()) {
                    if (outerWsMgr.hasConfiguration(type)) {
                        wsBuilder.with(type, outerWsMgr.getWorkspaceName(type), outerWsMgr.getConfiguration(type));
                    }
                }

                final LayerWorkspaceMgr wsMgr = wsBuilder
                        .with(ArrayType.FF_WORKING_MEM, "WS_ODE_VERTEX_LAYER_WORKING_MEM", WS_LAYER_WORKING_MEM_CONFIG)
                        .with(ArrayType.BP_WORKING_MEM, "WS_ODE_VERTEX_LAYER_WORKING_MEM", WS_LAYER_WORKING_MEM_CONFIG)
                        .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, "WS_ODE_VERTEX_RNN_LOOP_WORKING_MEM", WS_RNN_LOOP_WORKING_MEM_CONFIG)
                        .with(ArrayType.RNN_BP_LOOP_WORKING_MEM, "WS_ODE_VERTEX_RNN_LOOP_WORKING_MEM", WS_RNN_LOOP_WORKING_MEM_CONFIG)
                        .with(ArrayType.ACTIVATIONS, "WS_ODE_VERTEX_ALL_LAYERS_ACT", WS_ALL_LAYERS_ACT_CONFIG)
                        .with(ArrayType.ACTIVATION_GRAD, "WS_ODE_VERTEX_ALL_LAYERS_GRAD", WS_ALL_LAYERS_ACT_CONFIG)
                        .build();
                wsMgr.setHelperWorkspacePointers(outerWsMgr.getHelperWorkspacePointers());
                return wsMgr;
            }
        }.spyWsConfigs();
    }
}
