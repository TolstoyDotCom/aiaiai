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

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.Set;
import java.lang.invoke.MethodHandles;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.converters.ConverterUtils.DataSource;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.tolstoy.aiaiai.api.IKeyedInstance;
import com.tolstoy.aiaiai.api.AttributeNotSetException;

public class KeyedInstance<C,V> implements IKeyedInstance<C,V> {
	private static final Logger logger = LogManager.getLogger( MethodHandles.lookup().lookupClass() );

	private final Map<String,V> values;
	private C expectedClass;

	public KeyedInstance() throws Exception {
		this.values = new HashMap<String,V>();
	}

	public Set<String> getKeys() {
		return values.keySet();
	}

	public V getValue( String key ) {
		return values.get( key );
	}

	public C getExpectedClass() {
		return expectedClass;
	}

	public void setValue( String key, V value ) {
		values.put( key, value );
	}

	public void setExpectedClass( C expectedClass ) {
		this.expectedClass = expectedClass;
	}

	@Override
	public String toString() {
		String ret = expectedClass + " for: ";

		List<String> components = new ArrayList<String>();
		for ( String key : values.keySet() ) {
			components.add( key + "=" + values.get( key ) );
		}

		ret += StringUtils.join( components, "\t\t" );

		return ret;
	}
}
