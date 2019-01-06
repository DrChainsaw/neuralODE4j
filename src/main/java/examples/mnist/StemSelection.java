package examples.mnist;

import com.beust.jcommander.Parameter;

/**
 * Selects the stem of the network
 */
public class StemSelection {

    @Parameter(names = {"-downsampling-method", "-stem"}, description = "How to down sample the input, what is usually " +
            "referred to as the \"stem\" of a residual network. Supported options are \"conv\" and \"res\".")
    private Option stem = Option.res;

    public enum Option {
        res,
        conv
    }

    public Block get(int nrofKernels) {
       switch (stem) {
           case res: return new ResBlockStem(nrofKernels);
           case conv: return new ConvStem(nrofKernels);
           default: throw new IllegalArgumentException("Option not supported: " + stem + "!");
       }
    }

    /**
     * Return the name of the StemSelection
     * @return the name of the StemSelection
     */
    public String name() {
        return stem.name();
    }
}
