#include "Plan.pb.h"
#include "PlanBaseVisitor.h"
#include "PlanLexer.h"
#include "PlanParser.h"
#include <any>
#include <iostream>
#include <string>
namespace milvus {

struct ExprWithDtype {
  proto::plan::Expr *expr;
  proto::schema::DataType dtype;
  bool dependent;
  ExprWithDtype(proto::plan::Expr *const expr, proto::schema::DataType dtype,
                bool dependent)
      : expr(expr), dtype(dtype), dependent(dependent) {}
};

template <typename T> std::optional<T> extractValue(proto::plan::Expr *expr) {
  if (!expr->has_value_expr())
    return std::nullopt;

  if constexpr (std::is_same_v<T, std::int64_t>) {
    if (expr->value_expr().has_value() &&
        expr->value_expr().value().has_int64_val())
      return expr->value_expr().value().int64_val();
  }
  if constexpr (std::is_same_v<T, double>) {
    if (expr->value_expr().has_value() &&
        expr->value_expr().value().has_float_val())
      return expr->value_expr().value().float_val();
  }

  if constexpr (std::is_same_v<T, std::string>) {
    if (expr->value_expr().has_value() &&
        expr->value_expr().value().has_string_val())
      return expr->value_expr().value().string_val();
  }
  if constexpr (std::is_same_v<T, bool>) {
    if (expr->value_expr().has_value() &&
        expr->value_expr().value().has_bool_val())
      return expr->value_expr().value().bool_val();
  }

  return std::nullopt;
};

template <typename T> proto::plan::Expr *createValueExpr(const T val) {
  auto expr = new proto::plan::Expr();
  auto val_expr = new proto::plan::ValueExpr();

  auto value = new proto::plan::GenericValue();
  if constexpr (std::is_same_v<T, int64_t>)
    value->set_int64_val(val);
  else if constexpr (std::is_same_v<T, std::string>)
    value->set_string_val(val);
  else if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>)
    value->set_float_val(val);
  else if constexpr (std::is_same_v<T, bool>)
    value->set_bool_val(val);
  else
    static_assert(false);

  val_expr->set_allocated_value(value);
  expr->set_allocated_value_expr(val_expr);
  return expr;
}

template <proto::plan::BinaryExpr_BinaryOp T>
proto::plan::Expr *createBinExpr(proto::plan::Expr *left,
                                 proto::plan::Expr *right) {

  auto expr = new proto::plan::Expr();
  auto bin_expr = new proto::plan::BinaryExpr();
  bin_expr->set_op(T);
  bin_expr->set_allocated_left(left);
  bin_expr->set_allocated_right(right);
  return expr;
}

template <proto::plan::ArithOpType T>
proto::plan::Expr *createBinArithExpr(proto::plan::Expr *left,
                                      proto::plan::Expr *right) {

  auto expr = new proto::plan::Expr();
  auto bin_expr = new proto::plan::BinaryArithExpr();
  bin_expr->set_op(T);
  bin_expr->set_allocated_left(left);
  bin_expr->set_allocated_right(right);
  return expr;
}

bool arithmeticDtype(proto::schema::DataType type) {
  switch (type) {
  case proto::schema::DataType::Float:
    return true;
  case proto::schema::DataType::Double:
    return true;
  case proto::schema::DataType::Int8:
    return true;
  case proto::schema::DataType::Int16:
    return true;
  case proto::schema::DataType::Int32:
    return true;
  case proto::schema::DataType::Int64:
    return true;
  default:
    return false;
  }
}

proto::schema::DataType getArrayElementType(proto::plan::Expr *expr) {
  if (expr->has_column_expr()) {
    return expr->column_expr().info().data_type();
  }
  if (expr->has_value_expr() && expr->value_expr().has_value() &&
      expr->value_expr().value().has_array_val()) {
    return expr->value_expr().value().array_val().element_type();
  }

  return proto::schema::DataType::None;
}

bool canArithmeticDtype(proto::schema::DataType left_type,
                        proto::schema::DataType right_type) {

  if (left_type == proto::schema::DataType::JSON &&
      right_type == proto::schema::DataType::JSON)
    return false;
  if (left_type == proto::schema::DataType::JSON && arithmeticDtype(right_type))
    return true;
  if (arithmeticDtype(left_type) && right_type == proto::schema::DataType::JSON)
    return true;
  if (arithmeticDtype(left_type) && arithmeticDtype(right_type))
    return true;
  return false;
}

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
          proto::schema::DataType::None, false);
    case PlanParser::DIV:
      return ExprWithDtype(
          createBinArithExpr<proto::plan::ArithOpType::Div>(a, b),
          proto::schema::DataType::None, false);

    case PlanParser::MOD:

      return ExprWithDtype(
          createBinArithExpr<proto::plan::ArithOpType::Mod>(a, b),
          proto::schema::DataType::None, false);

    default:
      assert(false);
    }
  }

  virtual std::any
  visitIdentifier(PlanParser::IdentifierContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLike(PlanParser::LikeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitEquality(PlanParser::EqualityContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitReverseRange(PlanParser::ReverseRangeContext *ctx) override {
    return visitChildren(ctx);
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
