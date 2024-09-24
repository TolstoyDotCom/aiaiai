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

import com.tolstoy.aiaiai.api.IClassifierBuilderFactory;
import com.tolstoy.aiaiai.api.IClassifierBuilder;
import com.tolstoy.aiaiai.api.IClassifierParams;
import com.tolstoy.aiaiai.api.AttributeNotSetException;

class ClassifierBuilder<V> implements IClassifierBuilder<V> {
	private static final Logger logger = LogManager.getLogger( MethodHandles.lookup().lookupClass() );

	private final Classifier classifier;
	private final Filter filter;
	private final List<Attribute> valueAttributes;
	private final List<String> classAttributeOptions;
	private final Attribute classAttribute;
	private final int maxAttributeIndex;

	ClassifierBuilder( String classifierName, List<String> classifierArguments, String filterName, List<String> filterArguments, File inputFile ) throws Exception {
		String args[];

		args = classifierArguments != null ? classifierArguments.toArray( new String[ 0 ] ) : null;
		this.classifier = AbstractClassifier.forName( classifierName, args );

		args = filterArguments != null ? filterArguments.toArray( new String[ 0 ] ) : null;
		this.filter = (Filter) Class.forName( filterName ).getDeclaredConstructor().newInstance();
		if ( args != null && ( this.filter instanceof OptionHandler ) ) {
			( (OptionHandler) this.filter ).setOptions( args );
		}

		Instances rawInstances = new Instances( new BufferedReader( new FileReader( inputFile ) ) );

		InstancesHelper helper = new InstancesHelper( rawInstances );

		this.classAttribute = helper.getClassAttribute();
		this.classAttributeOptions = helper.getClassAttributeOptions();
		this.valueAttributes = helper.getValueAttributes();

		rawInstances.setClassIndex( this.classAttribute.index() );

		this.filter.setInputFormat( rawInstances );

		Instances filteredInstances = Filter.useFilter( rawInstances, this.filter );

		this.classifier.buildClassifier( filteredInstances );

		int tempMax = classAttribute.index();
		for ( Attribute attr : valueAttributes ) {
			tempMax = Math.max( tempMax, attr.index() );
		}

		this.maxAttributeIndex = tempMax;
	}

	public IClassifierParams createClassifierParams() {
		return new ClassifierParams( classAttribute, valueAttributes );
	}

	public String classify( IClassifierParams<V> params ) throws Exception {
		List<V> values = params.getList();

		List<Attribute> attrs = new ArrayList<Attribute>( maxAttributeIndex );
		for ( int i = 0; i <= maxAttributeIndex; i++ ) {
			attrs.add( i, null );
		}

		for ( Attribute attr : valueAttributes ) {
			attrs.set( attr.index(), new Attribute( attr.name() ) );
		}

		attrs.set( classAttribute.index(), new Attribute( classAttribute.name(), classAttributeOptions ) );

		Instances instances = new Instances( "whatever", (ArrayList<Attribute>) attrs, 0 );

		double[] attrValuesArray = new double[ values.size() ];
		int i = 0;
		for ( V val : values ) {
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

		Instances filteredInstances = Filter.useFilter( instances, filter );

		double rawPrediction = classifier.classifyInstance( filteredInstances.get( 0 ) );

		String prediction = instances.classAttribute().value( (int) rawPrediction );

		return prediction;
	}
}
