package vct.col.ast.util;

import static hre.lang.System.Fail;

/*
 *This class is used for reading .jspec files. In .jspec all the variables are declared globally and some are declared again in a rewrite rule. Hence requiring the use of shadow variables.
 */
public class SingleNameSpaceWithShadowVariables extends SingleNameSpace {

    @Override
    public void add(String name, VariableInfo def){
        map.put(name, def);
    }
}
