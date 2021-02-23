package vct.col.ast.util;

import vct.col.ast.expr.NameExpressionKind;
import vct.col.ast.generic.ASTNode;

/**
 * Information record for variables.
 *
 * @author Stefan Blom
 */
public class VariableInfo {

    /**
     * Reference to the place where the variable was defined.
     */
    public final ASTNode reference;

    /**
     * Stores the kind of the variable.
     */
    public final NameExpressionKind kind;

    /**
     * Constructor for a variable info record.
     */
    public VariableInfo(ASTNode reference, NameExpressionKind kind) {
        this.reference = reference;
        this.kind = kind;
    }
}
