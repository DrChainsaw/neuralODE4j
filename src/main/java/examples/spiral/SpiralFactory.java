package examples.spiral;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import util.plot.Plot;
import util.plot.RealTimePlot;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BooleanSupplier;
import java.util.function.DoubleSupplier;

/**
 * Creates spirals, equivalent to generate_spiral2d in https://github.com/rtqichen/torchdiffeq/blob/master/examples/latent_ode.py.
 * Formula for a spiral {@code r = a + b * theta} where theta is linearly spaced from a specified start and stop value.
 *
 * @author Christian Skarby
 */
class SpiralFactory {

    private final Spiral baseCw;
    private final Spiral baseCc;

    static class Spiral {
        private final INDArray trajectory;
        private final INDArray theta;

        private Spiral(INDArray trajectory, INDArray theta) {
            this.trajectory = trajectory;
            this.theta = theta;
        }

        void plot(Plot<Double, Double> plot, String series) {
            for(int i = 0; i < trajectory.size(1); i++) {
                plot.plotData(series, trajectory.getDouble(0, i), trajectory.getDouble(1, i));
            }
        }

        void plotBase(Plot<Double, Double> plot, String series) {
            plot(plot, series);
        }

        INDArray trajectory() {
            return trajectory;
        }

        INDArray theta() {
            return theta;
        }
    }

    static class SpiralFragment extends Spiral {

        private final Spiral base;

        public SpiralFragment(Spiral base, INDArray trajectory, INDArray theta) {
            super(trajectory, theta);
            this.base = base;
        }

        @Override
        void plotBase(Plot<Double, Double> plot, String series) {
            base.plotBase(plot, series);
        }
    }


    SpiralFactory(double a, double b, double startTheta, double stopTheta, long nrofSamples) {
        final INDArray thetaCw = Nd4j.linspace(startTheta, stopTheta, nrofSamples);
        final INDArray rCw = thetaCw.mul(b).addi(a);
        final INDArray trajectoryCw = Nd4j.vstack(rCw.mul(Transforms.cos(thetaCw)).add(5), rCw.mul(Transforms.sin(thetaCw)));
        this.baseCw = new Spiral(trajectoryCw, thetaCw);

        final INDArray thetaCc = thetaCw.rsub(1 + stopTheta);
        final INDArray rCc = thetaCc.rdiv(50).mul(b).addi(a);
        final INDArray trajectoryCc = Nd4j.vstack(rCc.mul(Transforms.cos(thetaCc)).sub(5), rCc.mul(Transforms.sin(thetaCc)));
        this.baseCc = new Spiral(trajectoryCc, thetaCc);
    }

    void plotClockWise(Plot<Double, Double> plot, String label) {
        baseCw.plot(plot, label);
    }

    void plotCounterClock(Plot<Double, Double> plot, String label) {
        baseCc.plot(plot, label);
    }

    List<Spiral> sample(long nrofSpirals, long nrofSamples, DoubleSupplier startSupplier, BooleanSupplier cwOrCc) {
        final List<Spiral> output = new ArrayList<>();
        for(int i = 0; i < nrofSpirals; i++) {
            Spiral base = cwOrCc.getAsBoolean() ? baseCw : baseCw;
            long start = (long)Math.min(base.theta.length() - nrofSamples, startSupplier.getAsDouble() * base.theta.length());

            output.add(new SpiralFragment(
                    base,
                    base.trajectory.get(NDArrayIndex.all(), NDArrayIndex.interval(start, start + nrofSamples)),
                    base.theta.get(NDArrayIndex.all(), NDArrayIndex.interval(start, start + nrofSamples)))
            );
        }
        return output;
    }


    public static void main(String[] args) {
        final Plot<Double, Double> plot = new RealTimePlot<>("Spiral test", "");
        final SpiralFactory factory = new SpiralFactory(0, 0.3, 0, 6 * Math.PI, 1000);

        final String cw = "ClockWise";
        final String cc = "CounterClock";
        plot.createSeries(cw);
        plot.createSeries(cc);
        factory.plotClockWise(plot, cw);
        factory.plotCounterClock(plot, cc);
    }

}
