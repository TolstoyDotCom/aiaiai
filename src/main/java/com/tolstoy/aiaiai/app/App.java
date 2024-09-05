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
import java.util.Map;
import java.util.HashMap;
import java.util.Set;
import java.util.Collections;
import java.util.Comparator;
import java.lang.invoke.MethodHandles;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.commons.lang3.StringUtils;
import weka.core.Instances;

import com.tolstoy.aiaiai.api.IClassifierBuilderFactory;
import com.tolstoy.aiaiai.api.IClassifierBuilder;
import com.tolstoy.aiaiai.api.IClassifierParams;
import com.tolstoy.aiaiai.api.IKeyedInstanceSet;
import com.tolstoy.aiaiai.api.IKeyedInstance;
import com.tolstoy.aiaiai.api.IConfidenceMatrix;

public class App {
	private static final Logger logger = LogManager.getLogger( MethodHandles.lookup().lookupClass() );

	private static final String CLASSIFIER_NAME = "weka.classifiers.trees.J48";
	//private static final String FILTER_NAME = "weka.filters.unsupervised.instance.Randomize";
	private static final String FILTER_NAME = "weka.filters.unsupervised.attribute.Normalize";

	public App() {
		try {
			List<String> classifierArguments = new ArrayList<String>();
			classifierArguments.add( "-C" );
			classifierArguments.add( "0.25" );
			classifierArguments.add( "-M" );
			classifierArguments.add( "2" );

			List<String> filterArguments = new ArrayList<String>();
			filterArguments.add( "-S" );
			filterArguments.add( "1.0" );
			filterArguments.add( "-T" );
			filterArguments.add( "0.0" );

			File inputFile = Paths.get( App.class.getResource( "/iris.arff" ).toURI() ).toFile();

			IClassifierBuilderFactory classifierBuilderFactory = new ClassifierBuilderFactory();

			IClassifierBuilder builder = classifierBuilderFactory.createClassifierBuilder( CLASSIFIER_NAME, classifierArguments, FILTER_NAME, filterArguments, inputFile );

			IClassifierParams params = builder.createClassifierParams();

			IKeyedInstanceSet keyedInstanceSet = new KeyedInstanceSet<String,Double>( inputFile );

			int numMatches = 0;

			List<IKeyedInstance<String,Double>> keyedInstances = keyedInstanceSet.getKeyedInstances();

			Map<String,Integer> numPerClass = countNumPerClass( keyedInstances );

			IConfidenceMatrix confidenceMatrix = classifierBuilderFactory.createConfidenceMatrix( numPerClass );

			countNumCorrectByClass( builder, params, keyedInstances, confidenceMatrix );

			logger.info( "RESULTS:\n\t" + StringUtils.join( confidenceMatrix.getResults(), "\n\t" ) );
		}
		catch ( Exception e ) {
			logger.catching( e );
		}
	}

	protected Map<String,Integer> countNumPerClass( List<IKeyedInstance<String,Double>> keyedInstances ) {
		Map<String,Integer> ret = new HashMap<String,Integer>();

		for ( IKeyedInstance<String,Double> keyedInstance : keyedInstances ) {
			String expected = keyedInstance.getExpectedClass();
			ret.put( expected, ret.getOrDefault( expected, 0 ) + 1 );
		}

		return ret;
	}

	protected void countNumCorrectByClass( IClassifierBuilder builder, IClassifierParams params, List<IKeyedInstance<String,Double>> keyedInstances, IConfidenceMatrix confidenceMatrix ) throws Exception {
		for ( IKeyedInstance<String,Double> keyedInstance : keyedInstances ) {
			String expected = keyedInstance.getExpectedClass();
			Set<String> keys = keyedInstance.getKeys();
			for ( String key : keys ) {
				params.setValue( key, keyedInstance.getValue( key ) );
			}

			String actual = builder.classify( params );

			confidenceMatrix.addPrediction( expected, actual );
		}
	}

	public static void main( String[] args ) {
		new App();
	}
}
