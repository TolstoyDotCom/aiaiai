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
import java.util.Map;
import java.util.HashMap;
import java.lang.invoke.MethodHandles;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.converters.ConverterUtils.DataSource;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.tolstoy.aiaiai.api.IKeyedInstanceSet;
import com.tolstoy.aiaiai.api.IKeyedInstance;
import com.tolstoy.aiaiai.api.AttributeNotSetException;

public class KeyedInstanceSet<C,V> implements IKeyedInstanceSet<C,V> {
	private static final Logger logger = LogManager.getLogger( MethodHandles.lookup().lookupClass() );

	private final List<IKeyedInstance<C,V>> keyedInstances;

	public KeyedInstanceSet( File inputFile ) throws Exception {
		Instances instances = DataSource.read( inputFile.getAbsolutePath() );

		InstancesHelper helper = new InstancesHelper( instances );

		List<Attribute> valueAttributes = helper.getValueAttributes();
		Attribute classAttribute = helper.getClassAttribute();

		this.keyedInstances = new ArrayList<IKeyedInstance<C,V>>( instances.numInstances() );

		instances.setClassIndex( classAttribute.index() );

		for ( Instance instance : instances ) {
			this.keyedInstances.add( createKeyedInstance( instance, valueAttributes, classAttribute ) );
		}
	}

	public List<IKeyedInstance<C,V>> getKeyedInstances() {
		return keyedInstances;
	}

	protected IKeyedInstance createKeyedInstance( Instance instance, List<Attribute> valueAttributes, Attribute classAttribute ) throws Exception {
		IKeyedInstance ret = new KeyedInstance<C,V>();

		for ( Attribute valueAttribute : valueAttributes ) {
			ret.setValue( valueAttribute.name(), instance.value( valueAttribute.index() ) );
		}

		ret.setExpectedClass( classAttribute.value( (int) instance.value( classAttribute.index() ) ) );

		return ret;
	}

	@Override
	public String toString() {
		String ret = keyedInstances.size() + " instances:\n";

		int count = 0;

		for ( IKeyedInstance keyedInstance : keyedInstances ) {
			ret += keyedInstance + "\n";
			if ( count++ > 10 ) {
				ret += "...\n";
				break;
			}
		}

		return ret;
	}
}
