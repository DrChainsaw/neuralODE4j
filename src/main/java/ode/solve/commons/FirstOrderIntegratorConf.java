package ode.solve.commons;

import ode.solve.api.FirstOrderSolver;
import ode.solve.api.FirstOrderSolverConf;
import ode.solve.conf.SolverConfig;
import org.apache.commons.math3.ode.FirstOrderIntegrator;
import org.apache.commons.math3.ode.nonstiff.DormandPrince54Integrator;

import java.util.HashMap;
import java.util.Map;

/**
 * Generic configuration for {@link FirstOrderIntegrator}s.
 *
 * @author Christian Skarby
 */
public class FirstOrderIntegratorConf implements FirstOrderSolverConf {

    private final static Map<String, Factory> factorymap = new HashMap<>();

        private final SolverConfig config;
    private final String integratorName;

    /**
     * Factory interface as {@link FirstOrderIntegrator}s are generally not serializable. Factories must be mapped to
     * integratorName through {@link #addIntegrator(String, Factory)} before instantiation.
     */
    public interface Factory {
        FirstOrderIntegrator create(SolverConfig config);
    }

    public FirstOrderIntegratorConf() {
        this(DormandPrince54Integrator.class.getName(), new SolverConfig(1e-3, 1e-3, 1e-10, 10));
    }

    public FirstOrderIntegratorConf(String integratorName, SolverConfig config) {
        this.integratorName = integratorName;
        this.config = config;
    }

    /**
     * Add a new factory to integrator mapping. Needs to be called before deserialization
     * @param name Name to associate the factory with
     * @param factory factory to produce the solver
     */
    public static void addIntegrator(String name, Factory factory) {
        factorymap.put(name, factory);
    }

    @Override
    public FirstOrderSolver instantiate() {
        final Factory factory = factorymap.get(integratorName);
        if(factory != null) {
            return new FirstOrderSolverAdapter(factory.create(config));
        }
        return new FirstOrderSolverAdapter(defaultCreate());
    }

    @Override
    public FirstOrderSolverConf clone() {
        return new FirstOrderIntegratorConf(integratorName, new SolverConfig(
                config.getAbsTol(),
                config.getRelTol(),
                config.getMinStep(),
                config.getMaxStep()));
    }

    private FirstOrderIntegrator defaultCreate() {
        try {
            return (FirstOrderIntegrator) Class.forName(integratorName).getConstructor(double.class, double.class, double.class, double.class)
            .newInstance(config.getMinStep(), config.getMaxStep(), config.getAbsTol(), config.getRelTol());
        } catch (Exception e) {
            throw new UnsupportedOperationException("Could not create " + integratorName + " from default signature!", e);
        }
    }
}
