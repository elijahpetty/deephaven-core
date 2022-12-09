package io.deephaven.engine.testutil.generator;

public class FromUniqueStringGenerator extends AbstractFromUniqueGenerator<String> {
    public FromUniqueStringGenerator(UniqueStringGenerator uniqueStringGenerator, double existingFraction) {
        this(uniqueStringGenerator, existingFraction, new StringGenerator());
    }

    FromUniqueStringGenerator(UniqueStringGenerator uniqueGenerator, double existingFraction,
            AbstractGenerator<String> defaultGenerator) {
        super(String.class, uniqueGenerator, defaultGenerator, String[]::new, existingFraction);
    }
}