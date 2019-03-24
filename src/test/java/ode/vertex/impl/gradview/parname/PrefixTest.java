package ode.vertex.impl.gradview.parname;

import org.junit.Test;
import org.nd4j.linalg.primitives.Pair;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link Prefix}
 *
 * @author Christian Skarby
 */
public class PrefixTest {

    /**
     * Test that a prefix is added
     */
    @Test
    public void map() {
        assertEquals("prefix_vertex-param", new Prefix("prefix_", new Concat()).map("vertex", "param"));
    }

    /**
     * Test that prefixing is reversed correctly
     */
    @Test
    public void reverseMap() {
        assertEquals(new Pair<>("vertex", "param"), new Prefix("prefix_", new Concat()).reverseMap("prefix_vertex-param"));
    }
}