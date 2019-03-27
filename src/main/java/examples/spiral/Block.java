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
     * @param builder Builder to add layers to
     * @param prev previous layers
     * @return name of last layer added
     */
    String add(ComputationGraphConfiguration.GraphBuilder builder, String... prev);
}
