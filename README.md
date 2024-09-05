# AIAIAI!

AIAIAI! is a small artificial intelligence (AI) library that might make dealing with Weka from Java easier.

Currently it assumes that the class attribute is a string and the other attributes are doubles. See iris.arff in the resources directory for such a setup.

Usage example:

```
File inputFile = new File( /*path to your arff file*/ );

IClassifierBuilderFactory classifierBuilderFactory = new ClassifierBuilderFactory();

IClassifierBuilder builder = classifierBuilderFactory.createClassifierBuilder( "weka.classifiers.trees.J48", null, "weka.filters.unsupervised.instance.Randomize", null, inputFile );

IClassifierParams params = builder.createClassifierParams();

params.setValue( "my_first_attribute", 1.1 );
params.setValue( "my_second_attribute", 2.2 );
params.setValue( "my_third_attribute", 3.3 );

String prediction = builder.classify( params );
```
