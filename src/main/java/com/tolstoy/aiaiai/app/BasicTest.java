/*
 * Copyright 2024 Chris Kelly
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

package com.tolstoy.aiaiai.app;

import java.nio.file.Paths;
import java.io.File;
import java.util.List;
import java.util.ArrayList;
import java.util.Enumeration;
import java.io.FileReader;
import java.io.BufferedReader;
import java.lang.invoke.MethodHandles;

import weka.classifiers.Classifier;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.DenseInstance;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.filters.Filter;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.tolstoy.aiaiai.api.IClassifierParams;
import com.tolstoy.aiaiai.api.AttributeNotSetException;

class BasicTest {
	private static final Logger logger = LogManager.getLogger( MethodHandles.lookup().lookupClass() );

	private final Classifier classifier;
	private final List<Attribute> valueAttributes;
	private final List<String> classAttributeOptions;
	private final Attribute classAttribute;

	private static final String CLASSIFIER_NAME = "weka.classifiers.trees.J48";
	//private static final String FILTER_NAME = "weka.filters.unsupervised.instance.Randomize";
	private static final String FILTER_NAME = "weka.filters.unsupervised.attribute.Normalize";
	private static final String FILE_NAME = "iris.arff";

	BasicTest() throws Exception {
		String args[];

		args = new String[]{ "-C", "0.25", "-M", "2" };
		this.classifier = AbstractClassifier.forName( CLASSIFIER_NAME, args );

		args = new String[]{ "-S", "1.0", "-T", "0.0" };
		Filter filter = (Filter) Class.forName( FILTER_NAME ).getDeclaredConstructor().newInstance();
		( (OptionHandler) filter ).setOptions( args );

		File inputFile = Paths.get( App.class.getResource( "/" + FILE_NAME ).toURI() ).toFile();

		Instances rawInstances = new Instances( new BufferedReader( new FileReader( inputFile ) ) );

		this.classAttribute = getClassAttribute( rawInstances );
		this.classAttributeOptions = getClassAttributeOptions( rawInstances );
		this.valueAttributes = getValueAttributes( rawInstances );

		rawInstances.setClassIndex( this.classAttribute.index() );

		filter.setInputFormat( rawInstances );

		Instances filteredInstances = Filter.useFilter( rawInstances, filter );

		this.classifier.buildClassifier( filteredInstances );

		int total = 0, numRight = 0;

		for ( Instance filteredInstance : filteredInstances ) {
			total++;

			List<Double> values = new ArrayList<Double>( 4 );
			for ( int i = 0; i < 4; i++ ) {
				values.add( filteredInstance.value( i ) );
			}
			String actual = filteredInstances.classAttribute().value( (int) filteredInstance.classValue() );
			String prediction = classify( values );
			if ( actual.equals( prediction ) ) {
				numRight++;
			}
		}

		logger.info( total + " " + numRight );
	}

	public String classify( List<Double> values ) throws Exception {
		List<Attribute> attrs = new ArrayList<Attribute>( values.size() + 1 );
		for ( Attribute attr : valueAttributes ) {
			attrs.add( attr.index(), new Attribute( attr.name() ) );
		}

		attrs.add( classAttribute.index(), new Attribute( classAttribute.name(), classAttributeOptions ) );

		Instances instances = new Instances( "whatever", (ArrayList<Attribute>) attrs, 0 );

		double[] attrValuesArray = new double[ values.size() ];
		int i = 0;
		for ( Double val : values ) {
			double casted;
			try {
				casted = Double.parseDouble( "" + val );
			}
			catch ( Exception e ) {
				casted = 0;
			}
			attrValuesArray[ i++ ] = casted;
		}

		Instance instance = new DenseInstance( 1.0, attrValuesArray );
		instances.add( instance );
		instance.setDataset( instances );
		instances.setClassIndex( classAttribute.index() );

		double rawPrediction = classifier.classifyInstance( instance );

		String prediction = instances.classAttribute().value( (int) rawPrediction );

		return prediction;
	}

	protected List<Attribute> getValueAttributes( Instances base ) {
		List<Attribute> ret = new ArrayList<Attribute>();

		Enumeration<Attribute> e = base.enumerateAttributes();
		while ( e.hasMoreElements() ) {
			Attribute a = e.nextElement();
			if ( a.enumerateValues() == null ) {
				ret.add( a );
			}
		}

		return ret;
	}

	protected Attribute getClassAttribute( Instances base ) {
		Enumeration<Attribute> e = base.enumerateAttributes();
		while ( e.hasMoreElements() ) {
			Attribute a = e.nextElement();
			if ( a.enumerateValues() != null ) {
				return a;
			}
		}

		throw new RuntimeException( "Class attribute not found" );
	}

	protected List<String> getClassAttributeOptions( Instances base ) {
		Attribute attr = getClassAttribute( base );

		Enumeration<Object> e = attr.enumerateValues();
		if ( e == null ) {
			throw new RuntimeException( "Class attribute has no options" );
		}

		List<String> ret = new ArrayList<String>();

		while ( e.hasMoreElements() ) {
			Object obj = e.nextElement();
			if ( obj instanceof String ) {
				ret.add( (String) obj );
			}
		}

		return ret;
	}

	public static void main( String[] args ) throws Exception {
		new BasicTest();
	}
}
