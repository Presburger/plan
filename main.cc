#include "Plan.pb.h"
#include "PlanBaseVisitor.h"
#include "PlanLexer.h"
#include "PlanParser.h"
#include <any>
#include <iostream>
#include <string>

#include "utils.h"

namespace milvus {

class PlanCCVisitor : public PlanVisitor {
public:
  // ok
  virtual std::any visitShift(PlanParser::ShiftContext *) override {
    assert(false);
    return nullptr;
  }
  // ok
  virtual std::any visitBitOr(PlanParser::BitOrContext *) override {
    assert(false);
    return nullptr;
  }
  // ok
  virtual std::any visitBitXor(PlanParser::BitXorContext *) override {
    assert(false);
    return nullptr;
  }
  // ok
  virtual std::any visitBitAnd(PlanParser::BitAndContext *) override {
    assert(false);
    return nullptr;
  }

  // ok
  virtual std::any visitParens(PlanParser::ParensContext *ctx) override {
    return visitChildren(ctx);
  }
  // ok
  virtual std::any visitString(PlanParser::StringContext *ctx) override {
    auto val = ctx->getText();
    return ExprWithDtype(createValueExpr<std::string>(val),
                         proto::schema::DataType::String, true);
  }
  // ok
  virtual std::any visitFloating(PlanParser::FloatingContext *ctx) override {
    auto text = ctx->getText();
    auto val = std::strtod(text.c_str(), NULL);
    return ExprWithDtype(createValueExpr<double>(val),
                         proto::schema::DataType::Float, true);
  }
  // ok
  virtual std::any visitInteger(PlanParser::IntegerContext *ctx) override {
    auto text = ctx->getText();
    int64_t val = std::strtoll(text.c_str(), NULL, 10);
    return ExprWithDtype(createValueExpr<int64_t>(val),
                         proto::schema::DataType::Int64, true);
  }
  // ok
  virtual std::any visitBoolean(PlanParser::BooleanContext *ctx) override {
    auto text = ctx->getText();
    bool val;
    std::istringstream(text) >> std::boolalpha >> val;
    return ExprWithDtype(createValueExpr<bool>(val),
                         proto::schema::DataType::Bool, true);
  }

  virtual std::any visitPower(PlanParser::PowerContext *ctx) override {
    auto a = std::any_cast<ExprWithDtype>(ctx->expr()[0]->accept(this)).expr;
    auto b = std::any_cast<ExprWithDtype>(ctx->expr()[1]->accept(this)).expr;

    float left, right;

    {
      auto opt_1 = extractValue<int64_t>(a);
      auto opt_2 = extractValue<double>(a);
      assert(opt_1.has_value() || opt_2.has_value());
      if (opt_1.has_value())
        left = float(opt_1.value());
      else
        left = float(opt_2.value());
    }

    {
      auto opt_1 = extractValue<int64_t>(b);
      auto opt_2 = extractValue<double>(b);
      assert(opt_1.has_value() || opt_2.has_value());
      if (opt_1.has_value())
        right = float(opt_1.value());
      else
        right = float(opt_2.value());
    }
    return ExprWithDtype(createValueExpr<double>(powf(left, right)),
                         proto::schema::DataType::Double, false);
  }

  virtual std::any visitLogicalOr(PlanParser::LogicalOrContext *ctx) override {
    auto a_expr_with_type =
        std::any_cast<ExprWithDtype>(ctx->expr()[0]->accept(this));
    auto b_expr_with_type =
        std::any_cast<ExprWithDtype>(ctx->expr()[1]->accept(this));

    auto a = a_expr_with_type.expr;
    auto b = b_expr_with_type.expr;

    if (extractValue<bool>(a).has_value() &&
        extractValue<bool>(b).has_value()) {
      return ExprWithDtype(
          createValueExpr<bool>(extractValue<bool>(a).value() ||
                                extractValue<bool>(b).value()),
          proto::schema::DataType::Bool, false);
    }

    assert(!a_expr_with_type.dependent);
    assert(!b_expr_with_type.dependent);
    assert(a_expr_with_type.dtype == proto::schema::DataType::Bool);
    assert(b_expr_with_type.dtype == proto::schema::DataType::Bool);
    return ExprWithDtype(
        createBinExpr<proto::plan::BinaryExpr_BinaryOp_LogicalOr>(a, b),
        proto::schema::DataType::Bool, false);
  }

  virtual std::any
  visitLogicalAnd(PlanParser::LogicalAndContext *ctx) override {

    auto a_expr_with_type =
        std::any_cast<ExprWithDtype>(ctx->expr()[0]->accept(this));
    auto b_expr_with_type =
        std::any_cast<ExprWithDtype>(ctx->expr()[1]->accept(this));

    auto a = a_expr_with_type.expr;
    auto b = b_expr_with_type.expr;

    if (extractValue<bool>(a).has_value() &&
        extractValue<bool>(b).has_value()) {
      return ExprWithDtype(
          createValueExpr<bool>(extractValue<bool>(a).value() &&
                                extractValue<bool>(b).value()),
          proto::schema::DataType::Bool, false);
    }

    assert(!a_expr_with_type.dependent);
    assert(!b_expr_with_type.dependent);
    assert(a_expr_with_type.dtype == proto::schema::DataType::Bool);
    assert(b_expr_with_type.dtype == proto::schema::DataType::Bool);
    return ExprWithDtype(
        createBinExpr<proto::plan::BinaryExpr_BinaryOp_LogicalAnd>(a, b),
        proto::schema::DataType::Bool, false);
  }

  virtual std::any
  visitJSONIdentifier(PlanParser::JSONIdentifierContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitJSONContainsAll(PlanParser::JSONContainsAllContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMulDivMod(PlanParser::MulDivModContext *ctx) override {
    auto a_expr_with_type =
        std::any_cast<ExprWithDtype>(ctx->expr()[0]->accept(this));
    auto b_expr_with_type =
        std::any_cast<ExprWithDtype>(ctx->expr()[1]->accept(this));
    auto a = a_expr_with_type.expr;
    auto b = b_expr_with_type.expr;
    if (extractValue<double>(a).has_value() &&
        extractValue<double>(b).has_value()) {
      switch (ctx->op->getType()) {
      case PlanParser::MUL:
        return ExprWithDtype(
            createValueExpr<double>(extractValue<double>(a).value() *
                                    extractValue<double>(b).value()),
            proto::schema::DataType::Double, false);
      case PlanParser::DIV:
        return ExprWithDtype(
            createValueExpr<double>(extractValue<double>(a).value() /
                                    extractValue<double>(b).value()),
            proto::schema::DataType::Double, false);
      default:
        assert(false);
      }
    }

    if (extractValue<int64_t>(a).has_value() &&
        extractValue<int64_t>(b).has_value()) {
      switch (ctx->op->getType()) {
      case PlanParser::MUL:
        return createValueExpr<int64_t>(extractValue<int64_t>(a).value() *
                                        extractValue<int64_t>(b).value());
      case PlanParser::DIV:
        return createValueExpr<int64_t>(extractValue<int64_t>(a).value() /
                                        extractValue<int64_t>(b).value());
      case PlanParser::MOD:
        return createValueExpr<int64_t>(extractValue<int64_t>(a).value() %
                                        extractValue<int64_t>(b).value());
      default:
        assert(false);
      }
    }

    if (extractValue<double>(a).has_value() &&
        extractValue<int64_t>(b).has_value()) {
      switch (ctx->op->getType()) {
      case PlanParser::MUL:
        return createValueExpr<double>(
            extractValue<double>(a).value() *
            double(extractValue<int64_t>(b).value()));
      case PlanParser::DIV:
        return createValueExpr<double>(
            extractValue<double>(a).value() /
            double(extractValue<int64_t>(b).value()));
      default:
        assert(false);
      }
    }

    if (extractValue<int64_t>(a).has_value() &&
        extractValue<double>(b).has_value()) {
      switch (ctx->op->getType()) {
      case PlanParser::MUL:
        return double(extractValue<int64_t>(a).value()) *
               extractValue<double>(b).value();
      case PlanParser::DIV:
        return double(extractValue<int64_t>(a).value()) /
               extractValue<double>(b).value();
      default:
        assert(false);
      }
    }

    if (a->has_column_expr()) {
      assert(a->column_expr().info().data_type() !=
             proto::schema::DataType::Array);
      assert(a->column_expr().info().nested_path_size() == 0);
    }

    if (b->has_column_expr()) {
      assert(b->column_expr().info().data_type() !=
             proto::schema::DataType::Array);
      assert(b->column_expr().info().nested_path_size() == 0);
    }

    if (a_expr_with_type.dtype == proto::schema::DataType::Array) {
      if (b_expr_with_type.dtype == proto::schema::DataType::Array)
        assert(
            canArithmeticDtype(getArrayElementType(a), getArrayElementType(b)));
      else if (arithmeticDtype(b_expr_with_type.dtype))
        assert(
            canArithmeticDtype(getArrayElementType(a), b_expr_with_type.dtype));
      else
        assert(false);
    }

    if (b_expr_with_type.dtype == proto::schema::DataType::Array) {
      if (arithmeticDtype(a_expr_with_type.dtype))
        assert(
            canArithmeticDtype(a_expr_with_type.dtype, getArrayElementType(b)));
      else
        assert(false);
    }

    if (arithmeticDtype(a_expr_with_type.dtype) &&
        arithmeticDtype(b_expr_with_type.dtype)) {
      assert(
          canArithmeticDtype(a_expr_with_type.dtype, b_expr_with_type.dtype));
    } else {
      assert(false);
    }

    switch (ctx->op->getType()) {
    case PlanParser::MUL:
      return ExprWithDtype(
          createBinArithExpr<proto::plan::ArithOpType::Mul>(a, b),
          calDataType(&a_expr_with_type, &b_expr_with_type), false);
    case PlanParser::DIV:
      return ExprWithDtype(
          createBinArithExpr<proto::plan::ArithOpType::Div>(a, b),
          calDataType(&a_expr_with_type, &b_expr_with_type), false);

    case PlanParser::MOD:

      return ExprWithDtype(
          createBinArithExpr<proto::plan::ArithOpType::Mod>(a, b),
          calDataType(&a_expr_with_type, &b_expr_with_type), false);

    default:
      assert(false);
    }
  }

  virtual std::any
  visitIdentifier(PlanParser::IdentifierContext *ctx) override {
    auto identifier = ctx->getText();
    auto &field = helper->GetFieldFromNameDefaultJSON(identifier);
    std::vector<std::string> nested_path;
    if (field.name() != identifier) {
      nested_path.push_back(identifier);
    }
    assert(!(field.data_type() == proto::schema::DataType::JSON &&
             nested_path.empty()));
    auto expr = new proto::plan::Expr();
    auto col_expr = new proto::plan::ColumnExpr();
    auto info = new proto::plan::ColumnInfo();
    info->set_field_id(field.fieldid());
    info->set_data_type(field.data_type());
    info->set_is_primary_key(field.is_primary_key());
    info->set_is_autoid(field.autoid());
    for (int i = 0; i < (int)nested_path.size(); ++i)
      info->set_nested_path(i, nested_path[i]);
    info->set_is_primary_key(field.is_primary_key());
    info->set_element_type(field.element_type());
    col_expr->set_allocated_info(info);
    expr->set_allocated_column_expr(col_expr);
    return ExprWithDtype(expr, field.data_type(), false);
  }

  virtual std::any visitLike(PlanParser::LikeContext *ctx) override {
    auto a = std::any_cast<ExprWithDtype>(ctx->expr()->accept(this));
    auto a_expr = a.expr;
    assert(a_expr);
    auto info = a_expr->column_expr().info();
    assert(!(info.data_type() == proto::schema::DataType::JSON &&
             info.nested_path_size() == 0));
    assert((a.dtype == proto::schema::DataType::String ||
            a.dtype == proto::schema::DataType::JSON) ||
           (a.dtype == proto::schema::DataType::Array &&
            info.element_type() == proto::schema::DataType::String));

    auto str = ctx->StringLiteral()->getText();
    auto pattern = convertEscapeSingle(str);

    auto res = translatePatternMatch(pattern);

    auto expr = new proto::plan::Expr();
    auto unaryrange_expr = new proto::plan::UnaryRangeExpr();
    unaryrange_expr->set_op(res.first);
    unaryrange_expr->set_allocated_column_info(
        new proto::plan::ColumnInfo(info));
    expr->set_allocated_unary_range_expr(unaryrange_expr);
    return ExprWithDtype(expr, proto::schema::DataType::Bool, false);
  }

  virtual std::any visitEquality(PlanParser::EqualityContext *ctx) override {
    auto a = std::any_cast<ExprWithDtype>(ctx->expr()[0]->accept(this));
    auto b = std::any_cast<ExprWithDtype>(ctx->expr()[1]->accept(this));
    {
      auto a_opt = extractValue<bool>(a.expr);
      auto b_opt = extractValue<bool>(b.expr);
      if (a_opt.has_value() && b_opt.has_value()) {
      }
      switch (ctx->op->getType()) {
      case PlanParser::EQ:
        return createValueExpr<bool>(a_opt.value() == b_opt.value());
      case PlanParser::NE:
        return createValueExpr<bool>(a_opt.value() != b_opt.value());
      default:
        assert(false);
      }
    }

    {
      auto a_opt = extractValue<std::string>(a.expr);
      auto b_opt = extractValue<std::string>(b.expr);
      if (a_opt.has_value() && b_opt.has_value()) {
      }
      switch (ctx->op->getType()) {
      case PlanParser::EQ:
        return createValueExpr<bool>(a_opt.value() == b_opt.value());
      case PlanParser::NE:
        return createValueExpr<bool>(a_opt.value() != b_opt.value());
      default:
        assert(false);
      }
    }

    {
      auto a_opt = extractValue<double>(a.expr);
      auto b_opt = extractValue<int64_t>(b.expr);
      if (a_opt.has_value() && b_opt.has_value()) {
      }
      switch (ctx->op->getType()) {
      case PlanParser::EQ:
        return createValueExpr<bool>(a_opt.value() == b_opt.value());
      case PlanParser::NE:
        return createValueExpr<bool>(a_opt.value() != b_opt.value());
      default:
        assert(false);
      }
    }

    {
      auto a_opt = extractValue<int64_t>(a.expr);
      auto b_opt = extractValue<double>(b.expr);
      if (a_opt.has_value() && b_opt.has_value()) {
      }
      switch (ctx->op->getType()) {
      case PlanParser::EQ:
        return createValueExpr<bool>(a_opt.value() == b_opt.value());
      case PlanParser::NE:
        return createValueExpr<bool>(a_opt.value() != b_opt.value());
      default:
        assert(false);
      }
    }
    {
      auto a_opt = extractValue<int32_t>(a.expr);
      auto b_opt = extractValue<double>(b.expr);
      if (a_opt.has_value() && b_opt.has_value()) {
      }
      switch (ctx->op->getType()) {
      case PlanParser::EQ:
        return createValueExpr<bool>(a_opt.value() == b_opt.value());
      case PlanParser::NE:
        return createValueExpr<bool>(a_opt.value() != b_opt.value());
      default:
        assert(false);
      }
    }
    {
      auto a_opt = extractValue<double>(a.expr);
      auto b_opt = extractValue<int32_t>(b.expr);
      if (a_opt.has_value() && b_opt.has_value()) {
      }
      switch (ctx->op->getType()) {
      case PlanParser::EQ:
        return createValueExpr<bool>(a_opt.value() == b_opt.value());
      case PlanParser::NE:
        return createValueExpr<bool>(a_opt.value() != b_opt.value());
      default:
        assert(false);
      }
    }

    {
      auto a_opt = extractValue<float>(a.expr);
      auto b_opt = extractValue<double>(b.expr);
      if (a_opt.has_value() && b_opt.has_value()) {
      }
      switch (ctx->op->getType()) {
      case PlanParser::EQ:
        return createValueExpr<bool>(a_opt.value() == b_opt.value());
      case PlanParser::NE:
        return createValueExpr<bool>(a_opt.value() != b_opt.value());
      default:
        assert(false);
      }
    }
    {
      auto a_opt = extractValue<double>(a.expr);
      auto b_opt = extractValue<float>(b.expr);
      if (a_opt.has_value() && b_opt.has_value()) {
      }
      switch (ctx->op->getType()) {
      case PlanParser::EQ:
        return createValueExpr<bool>(a_opt.value() == b_opt.value());
      case PlanParser::NE:
        return createValueExpr<bool>(a_opt.value() != b_opt.value());
      default:
        assert(false);
      }
    }
    {
      auto a_opt = extractValue<int32_t>(a.expr);
      auto b_opt = extractValue<float>(b.expr);
      if (a_opt.has_value() && b_opt.has_value()) {
      }
      switch (ctx->op->getType()) {
      case PlanParser::EQ:
        return createValueExpr<bool>(a_opt.value() == b_opt.value());
      case PlanParser::NE:
        return createValueExpr<bool>(a_opt.value() != b_opt.value());
      default:
        assert(false);
      }
    }

    {
      auto a_opt = extractValue<float>(a.expr);
      auto b_opt = extractValue<int32_t>(b.expr);
      if (a_opt.has_value() && b_opt.has_value()) {
      }
      switch (ctx->op->getType()) {
      case PlanParser::EQ:
        return createValueExpr<bool>(a_opt.value() == b_opt.value());
      case PlanParser::NE:
        return createValueExpr<bool>(a_opt.value() != b_opt.value());
      default:
        assert(false);
      }
    }

    {
      auto a_opt = extractValue<int32_t>(a.expr);
      auto b_opt = extractValue<int32_t>(b.expr);
      if (a_opt.has_value() && b_opt.has_value()) {
      }
      switch (ctx->op->getType()) {
      case PlanParser::EQ:
        return createValueExpr<bool>(a_opt.value() == b_opt.value());
      case PlanParser::NE:
        return createValueExpr<bool>(a_opt.value() != b_opt.value());
      default:
        assert(false);
      }
    }

    {
      auto a_opt = extractValue<int32_t>(a.expr);
      auto b_opt = extractValue<uint64_t>(b.expr);
      if (a_opt.has_value() && b_opt.has_value()) {
      }
      switch (ctx->op->getType()) {
      case PlanParser::EQ:
        return createValueExpr<bool>(a_opt.value() == b_opt.value());
      case PlanParser::NE:
        return createValueExpr<bool>(a_opt.value() != b_opt.value());
      default:
        assert(false);
      }
    }

    {
      auto a_opt = extractValue<int32_t>(a.expr);
      auto b_opt = extractValue<uint64_t>(b.expr);
      if (a_opt.has_value() && b_opt.has_value()) {
      }
      switch (ctx->op->getType()) {
      case PlanParser::EQ:
        return createValueExpr<bool>(a_opt.value() == b_opt.value());
      case PlanParser::NE:
        return createValueExpr<bool>(a_opt.value() != b_opt.value());
      default:
        assert(false);
      }
    }

    proto::plan::OpType op;
    switch (ctx->op->getType()) {
    case PlanParser::LT:
      op = proto::plan::OpType::LessThan;
    case PlanParser::LE:
      op = proto::plan::OpType::LessEqual;
    case PlanParser::GT:
      op = proto::plan::OpType::GreaterThan;
    case PlanParser::GE:
      op = proto::plan::OpType::GreaterEqual;
    case PlanParser::EQ:
      op = proto::plan::OpType::Equal;
    case PlanParser::NE:
      op = proto::plan::OpType::NotEqual;
    }

    if (a.expr->has_value_expr() && !b.expr->has_value_expr()) {
      ExprWithDtype left = toValueExpr(
          new proto::plan::GenericValue(a.expr->value_expr().value()));
      ExprWithDtype right = b;

      return ExprWithDtype(HandleCompare(op, left, right),
                           proto::schema::DataType::Bool, false);
    }

    if (!a.expr->has_value_expr() && b.expr->has_value_expr()) {

      ExprWithDtype left = a;
      ExprWithDtype right = toValueExpr(
          new proto::plan::GenericValue(b.expr->value_expr().value()));

      return ExprWithDtype(HandleCompare(op, left, right),
                           proto::schema::DataType::Bool, false);
    }

    if (a.expr->has_value_expr() && b.expr->has_value_expr()) {

      ExprWithDtype left = toValueExpr(
          new proto::plan::GenericValue(a.expr->value_expr().value()));
      ExprWithDtype right = toValueExpr(
          new proto::plan::GenericValue(b.expr->value_expr().value()));

      return ExprWithDtype(HandleCompare(op, left, right),
                           proto::schema::DataType::Bool, false);
    }

    if (!a.expr->has_value_expr() && !b.expr->has_value_expr()) {

      return ExprWithDtype(HandleCompare(op, a, b),
                           proto::schema::DataType::Bool, false);
    }
  }

  proto::plan::ColumnInfo *
  getChildColumnInfo(antlr4::tree::TerminalNode *identifier,
                     antlr4::tree::TerminalNode *child) {
    if (identifier) {

      auto text = identifier->getText();
      auto &field = helper->GetFieldFromNameDefaultJSON(text);
      std::vector<std::string> nested_path;
      if (field.name() != text) {
        nested_path.push_back(text);
      }
      assert(!(field.data_type() == proto::schema::DataType::JSON &&
               nested_path.empty()));
      auto info = new proto::plan::ColumnInfo();
      info->set_field_id(field.fieldid());
      info->set_data_type(field.data_type());
      info->set_is_primary_key(field.is_primary_key());
      info->set_is_autoid(field.autoid());
      for (int i = 0; i < (int)nested_path.size(); ++i)
        info->set_nested_path(i, nested_path[i]);
      info->set_is_primary_key(field.is_primary_key());
      info->set_element_type(field.element_type());
      return info;
    }
  }

  virtual std::any
  visitReverseRange(PlanParser::ReverseRangeContext *ctx) override {
    ctx->In
  }

  virtual std::any visitAddSub(PlanParser::AddSubContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitRelational(PlanParser::RelationalContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitArrayLength(PlanParser::ArrayLengthContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTerm(PlanParser::TermContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitJSONContains(PlanParser::JSONContainsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRange(PlanParser::RangeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnary(PlanParser::UnaryContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitArray(PlanParser::ArrayContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitJSONContainsAny(PlanParser::JSONContainsAnyContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitExists(PlanParser::ExistsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitEmptyTerm(PlanParser::EmptyTermContext *ctx) override {
    return visitChildren(ctx);
  }

private:
  SchemaHelper *helper;
};

} // namespace milvus

int main(int, const char *argv[]) {

  std::string exprstr(argv[1]);
  antlr4::ANTLRInputStream input(exprstr);
  PlanLexer lexer(&input);
  antlr4::CommonTokenStream tokens(&lexer);
  PlanParser parser(&tokens);

  PlanParser::ExprContext *tree = parser.expr();
  milvus::PlanCCVisitor visitor;
  visitor.visit(tree);

  return 0;
}
