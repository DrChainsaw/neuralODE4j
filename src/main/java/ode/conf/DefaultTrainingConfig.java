package ode.conf;

import org.deeplearning4j.nn.api.TrainingConfig;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.nd4j.linalg.learning.config.IUpdater;

/**
 * Basic {@link TrainingConfig} for vertices which do not have a {@link org.deeplearning4j.nn.conf.layers.Layer}
 *
 * @author Christian Skarby
 */
public class DefaultTrainingConfig implements TrainingConfig {


    private final String name;
    private final IUpdater updater;

    public DefaultTrainingConfig(String name, IUpdater updater) {
        this.name = name;
        this.updater = updater;
    }

    @Override
    public String getLayerName() {
        return name;
    }

    @Override
    public boolean isPretrain() {
        return false;
    }

    @Override
    public double getL1ByParam(String paramName) {
        return 0;
    }

    @Override
    public double getL2ByParam(String paramName) {
        return 0;
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return false;
    }

    @Override
    public IUpdater getUpdaterByParam(String paramName) {
        return updater;
    }

    @Override
    public GradientNormalization getGradientNormalization() {
        return GradientNormalization.None;
    }

    @Override
    public double getGradientNormalizationThreshold() {
        return 0;
    }

    @Override
    public void setPretrain(boolean pretrain) {

    }
}
