package examples.anode;

/**
 * A simple block of layers
 *
 * @author Christian Skarby
 */
interface Block {

    /**
     * Add layers to given builder
     * @param builder
     * @param prev previous layers to be input to this block
     * @return name of last layer added
     */
    String add(GraphBuilderWrapper builder, String ... prev);
}
