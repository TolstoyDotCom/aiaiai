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
import java.util.Map;
import java.lang.invoke.MethodHandles;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.tolstoy.aiaiai.api.IClassifierBuilder;
import com.tolstoy.aiaiai.api.IClassifierBuilderFactory;
import com.tolstoy.aiaiai.api.IConfidenceMatrix;

public class ClassifierBuilderFactory implements IClassifierBuilderFactory {
	private static final Logger logger = LogManager.getLogger( MethodHandles.lookup().lookupClass() );

	public IClassifierBuilder createClassifierBuilder( String classifierName, List<String> classifierArguments, String filterName, List<String> filterArguments, File inputFile ) throws Exception {
		return new ClassifierBuilder( classifierName, classifierArguments, filterName, filterArguments, inputFile );
	}

	public IConfidenceMatrix createConfidenceMatrix( Map<String,Integer> classCounts ) {
		return new ConfidenceMatrix( classCounts );
	}
}
