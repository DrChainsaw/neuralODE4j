package examples.spiral;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;

/**
 * A simple block of layers
 *
 * @author Christian Skarby
 */
interface Block {

    /**
     * Add layers to given builder
     * @param builder
     * @return name of last layer added
     */
    String add(String prev, ComputationGraphConfiguration.GraphBuilder builder);
}
