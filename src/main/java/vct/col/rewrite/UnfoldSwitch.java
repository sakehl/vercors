package vct.col.rewrite;

import vct.col.ast.expr.NameExpression;
import vct.col.ast.expr.StandardOperator;
import vct.col.ast.generic.ASTNode;
import vct.col.ast.stmt.composite.*;
import vct.col.ast.stmt.decl.ProgramUnit;
import vct.col.ast.util.AbstractRewriter;

import java.util.Objects;
import java.util.stream.Collectors;

public class UnfoldSwitch extends AbstractRewriter {
    int counter = 0;

    public UnfoldSwitch(ProgramUnit source) {
        super(source);
    }

    public NameExpression generateCaseIDLabel(String switchID, int caseID) {
        return generateCaseIDLabel(switchID, caseID + "");
    }

    public NameExpression generateCaseIDLabel(String switchID, String suffix) {
        return create.label(switchID + "_case_" + suffix);
    }

    public void visit(Switch switchStatement) {
        super.visit(switchStatement);

        BlockStatement mainBlock = create.block();
        BlockStatement caseStatementsBlock = create.block();
        String switchID = Objects.requireNonNull(switchStatement.getLabel(0).getName());

        // Put the case expr in a var s.t. it can be referenced in the if chain
        String exprName = switchID + "_" + counter++;
        mainBlock.add(create.field_decl(exprName, switchStatement.expr.getType(), switchStatement.expr));

        // Create if chain jumping to all the numbered arms
        IfStatement ifChain = null;
        boolean encounteredDefault = false;
        for (int caseID = 0; caseID < switchStatement.cases.length; caseID++) {
            Switch.Case switchCase = switchStatement.cases[caseID];

            if (switchCase.cases.contains(null)) {
                // Switch case contains default label
                encounteredDefault = true;
                caseStatementsBlock.add(create.labelDecl(generateCaseIDLabel(switchID, "default")));
            }

            // Fold all "switchValue == labelValue" expressions with "||", skipping default case labels (null)
            ASTNode ifGuard = create.fold(StandardOperator.Or,
                    switchCase.cases.stream()
                        .filter(Objects::nonNull)
                        .map(caseExpr -> eq(name(exprName), caseExpr))
                        .collect(Collectors.toList()));

            // Add if to the chain that jumps to the case statements
            NameExpression caseIDLabel = generateCaseIDLabel(switchID, caseID);
            // If there was only a default case there is no if guard, so we add no if.
            // The if for default will be added at the end, after all the other are added
            if (ifGuard != null) {
                IfStatement nextIf = create.ifthenelse(ifGuard, create.gotoStatement(caseIDLabel));
                if (ifChain == null) {
                    ifChain = nextIf;
                    mainBlock.add(ifChain);
                } else {
                    ifChain.addClause(IfStatement.elseGuard(), nextIf);
                    ifChain = nextIf;
                }
            }

            // TODO: This breaks for nested switches, since we do not call rewrite on the case statements
            // TODO: Refactor cases to be their own respective AST nodes? instead of doing them manually like this
            caseStatementsBlock.add(create.labelDecl(caseIDLabel));
            for (ASTNode caseStatement : switchCase.stats) {
                caseStatementsBlock.add(caseStatement);
            }
        }

        if (encounteredDefault) {
            ASTNode defaultJump = create.gotoStatement(generateCaseIDLabel(switchID, "default"));
            if (ifChain == null) {
                // No other cases, just the default case
                mainBlock.add(defaultJump);
            } else {
                // Other cases, add the default case as a last case
                ifChain.addClause(IfStatement.elseGuard(), defaultJump);
            }
        }

        // Add all the statements for the cases at the end, when it's sure that the if-chain is added
        mainBlock.add(caseStatementsBlock);

        result = mainBlock;
    }
}
