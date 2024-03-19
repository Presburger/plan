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
    return ExprWithDtype(createValueExpr<std::string>(val, this->arena.get()),
                         proto::schema::DataType::String, true);
  }
  // ok
  virtual std::any visitFloating(PlanParser::FloatingContext *ctx) override {
    auto text = ctx->getText();
    auto val = std::strtod(text.c_str(), NULL);
    return ExprWithDtype(createValueExpr<double>(val, this->arena.get()),
                         proto::schema::DataType::Float, true);
  }
  // ok
  virtual std::any visitInteger(PlanParser::IntegerContext *ctx) override {
    auto text = ctx->getText();
    int64_t val = std::strtoll(text.c_str(), NULL, 10);
    return ExprWithDtype(createValueExpr<int64_t>(val, this->arena.get()),
                         proto::schema::DataType::Int64, true);
  }
  // ok
  virtual std::any visitBoolean(PlanParser::BooleanContext *ctx) override {
    auto text = ctx->getText();
    bool val;
    std::istringstream(text) >> std::boolalpha >> val;
    return ExprWithDtype(createValueExpr<bool>(val, this->arena.get()),
                         proto::schema::DataType::Bool, true);
  }

  virtual std::any visitPower(PlanParser::PowerContext *ctx) override {
    auto left_expr =
        std::any_cast<ExprWithDtype>(ctx->expr()[0]->accept(this)).expr;
    auto right_expr =
        std::any_cast<ExprWithDtype>(ctx->expr()[1]->accept(this)).expr;

    auto left = extractValue(left_expr);
    auto right = extractValue(right_expr);

    assert(left.has_value() && right.has_value());
    assert(left.type() == typeid(double) || left.type() == typeid(int64_t));
    assert(right.type() == typeid(double) || right.type() == typeid(int64_t));
    float left_value, right_value;
    if (left.type() == typeid(int64_t))
      left_value = float(std::any_cast<int64_t>(left));
    if (left.type() == typeid(double))
      left_value = float(std::any_cast<double>(left));
    if (right.type() == typeid(int64_t))
      right_value = float(std::any_cast<int64_t>(right));
    if (right.type() == typeid(double))
      right_value = float(std::any_cast<double>(right));

    return ExprWithDtype(createValueExpr<double>(powf(left_value, right_value),
                                                 this->arena.get()),
                         proto::schema::DataType::Double, false);
  }

  virtual std::any visitLogicalOr(PlanParser::LogicalOrContext *ctx) override {
    auto left_expr_with_type =
        std::any_cast<ExprWithDtype>(ctx->expr()[0]->accept(this));
    auto right_expr_with_type =
        std::any_cast<ExprWithDtype>(ctx->expr()[1]->accept(this));

    auto left_expr = left_expr_with_type.expr;
    auto right_expr = right_expr_with_type.expr;

    auto left_value = extractValue(left_expr);
    auto right_value = extractValue(right_expr);

    if (left_value.has_value() && right_value.has_value() &&
        left_value.type() == typeid(bool) &&
        right_value.type() == typeid(bool)) {
      return ExprWithDtype(
          createValueExpr<bool>(std::any_cast<bool>(left_value) ||
                                    std::any_cast<bool>(right_value),
                                this->arena.get()

                                    ),
          proto::schema::DataType::Bool, false

      )
    }

    assert(!left_expr_with_type.dependent);
    assert(!right_expr_with_type.dependent);
    assert(left_expr_with_type.dtype == proto::schema::DataType::Bool);
    assert(right_expr_with_type.dtype == proto::schema::DataType::Bool);
    return ExprWithDtype(
        createBinExpr<proto::plan::BinaryExpr_BinaryOp_LogicalOr>(left_expr,
                                                                  right_expr),
        proto::schema::DataType::Bool, false);
  }

  virtual std::any
  visitLogicalAnd(PlanParser::LogicalAndContext *ctx) override {

    auto left_expr_with_type =
        std::any_cast<ExprWithDtype>(ctx->expr()[0]->accept(this));
    auto right_expr_with_type =
        std::any_cast<ExprWithDtype>(ctx->expr()[1]->accept(this));

    auto left_expr = left_expr_with_type.expr;
    auto right_expr = right_expr_with_type.expr;

    auto left_value = extractValue(left_expr);
    auto right_value = extractValue(right_expr);

    if (left_value.has_value() && right_value.has_value() &&
        left_value.type() == typeid(bool) &&
        right_value.type() == typeid(bool)) {
      return ExprWithDtype(
          createValueExpr<bool>(std::any_cast<bool>(left_value) ||
                                    std::any_cast<bool>(right_value),
                                this->arena.get()

                                    ),
          proto::schema::DataType::Bool, false

      );
    }

    assert(!left_expr_with_type.dependent);
    assert(!right_expr_with_type.dependent);
    assert(left_expr_with_type.dtype == proto::schema::DataType::Bool);
    assert(right_expr_with_type.dtype == proto::schema::DataType::Bool);
    return ExprWithDtype(
        createBinExpr<proto::plan::BinaryExpr_BinaryOp_LogicalOr>(left_expr,
                                                                  right_expr),
        proto::schema::DataType::Bool, false);
  }

  virtual std::any
  visitJSONIdentifier(PlanParser::JSONIdentifierContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitJSONContainsAll(PlanParser::JSONContainsAllContext *ctx) override {
    auto field = std::any_cast<ExprWithDtype>(ctx->expr()[0]->accept(this));
    auto info = field.expr->column_expr().info();
    assert(info.data_type() == proto::schema::DataType::Array ||
           info.data_type() == proto::schema::DataType::JSON);
    auto elem = std::any_cast<ExprWithDtype>(ctx->expr()[1]->accept(this));
    if (info.data_type() == proto::schema::DataType::Array) {
      proto::plan::GenericValue expr =
          proto::plan::GenericValue(elem.expr->value_expr().value());
      assert(canBeCompared(field, toValueExpr(&expr)));
    }

    auto expr = google::protobuf::Arena::CreateMessage<proto::plan::Expr>(
        this->arena.get());
    auto json_contain_expr =
        google::protobuf::Arena::CreateMessage<proto::plan::JSONContainsExpr>(
            this->arena.get());
    auto value = json_contain_expr->add_elements(); // MayBe BUG
    value->unsafe_arena_set_allocated_array_val(
        google::protobuf::Arena::CreateMessage<proto::plan::Array>(
            this->arena.get(), elem.expr->value_expr().value().array_val()));
    json_contain_expr->set_elements_same_type(
        elem.expr->value_expr().value().array_val().same_type());
    json_contain_expr->unsafe_arena_set_allocated_column_info(
        google::protobuf::Arena::CreateMessage<proto::plan::ColumnInfo>(
            this->arena.get(), info));

    json_contain_expr->set_op(proto::plan::JSONContainsExpr_JSONOp_ContainsAll);
    expr->unsafe_arena_set_allocated_json_contains_expr(json_contain_expr);
    return ExprWithDtype(expr, proto::schema::Bool, false);
  }

  virtual std::any visitMulDivMod(PlanParser::MulDivModContext *ctx) override {
    auto left_expr_with_type =
        std::any_cast<ExprWithDtype>(ctx->expr()[0]->accept(this));
    auto right_expr_with_type =
        std::any_cast<ExprWithDtype>(ctx->expr()[1]->accept(this));
    auto left_expr = left_expr_with_type.expr;
    auto right_expr = right_expr_with_type.expr;

    auto left_value = extractValue(left_expr);
    auto right_value = extractValue(right_expr);
    if (left_value.has_value() && right_value.has_value()) {

      if (left_value.type() == typeid(double) &&
          right_value.type() == typeid(double)) {
        switch (ctx->op->getType()) {
        case PlanParser::MUL:
          return ExprWithDtype(
              createValueExpr<double>(std::any_cast<double>(left_value) *
                                          std::any_cast<double>(right_value),
                                      this->arena.get()),
              proto::schema::DataType::Double, false);
        case PlanParser::DIV:
          return ExprWithDtype(
              createValueExpr<double>(std::any_cast<double>(left_value) /
                                          std::any_cast<double>(right_value),
                                      this->arena.get()),
              proto::schema::DataType::Double, false);
        default:
          assert(false);
        }
      }

      if (left_value.type() == typeid(int64_t) &&
          right_value.type() == typeid(int64_t)) {
        switch (ctx->op->getType()) {
        case PlanParser::MUL:
          return createValueExpr<int64_t>(
              std::any_cast<int64_t>(left_value) *
                  std::any_cast<int64_t>(right_value),
              this->arena.get());
        case PlanParser::DIV:
          return createValueExpr<int64_t>(
              std::any_cast<int64_t>(left_value) /
                  std::any_cast<int64_t>(right_value),
              this->arena.get());
        case PlanParser::MOD:
          return createValueExpr<int64_t>(
              std::any_cast<int64_t>(left_value) %
                  std::any_cast<int64_t>(right_value),
              this->arena.get());
        default:
          assert(false);
        }
      }

      if (left_value.type() == typeid(double) &&
          right_value.type() == typeid(int64_t)) {
        switch (ctx->op->getType()) {
        case PlanParser::MUL:
          return createValueExpr<double>(
              std::any_cast<double>(left_value) *
                  std::any_cast<int64_t>(right_value),
              this->arena.get());
        case PlanParser::DIV:
          return createValueExpr<double>(
              std::any_cast<double>(left_value) /
                  std::any_cast<int64_t>(right_value),
              this->arena.get());
        default:
          assert(false);
        }
      }

      if (left_value.type() == typeid(int64_t) &&
          right_value.type() == typeid(double)) {

        switch (ctx->op->getType()) {
        case PlanParser::MUL:
          return createValueExpr<double>(
              double(std::any_cast<int64_t>(left_value)) *
                  std::any_cast<double>(right_value),
              this->arena.get());
        case PlanParser::DIV:
          return createValueExpr<double>(
              double(std::any_cast<int64_t>(left_value)) *
                  std::any_cast<double>(right_value),
              this->arena.get());
        default:
          assert(false);
        }
      }
    }

    if (left_expr->has_column_expr()) {
      assert(left_expr->column_expr().info().data_type() !=
             proto::schema::DataType::Array);
      assert(left_expr->column_expr().info().nested_path_size() == 0);
    }

    if (right_expr->has_column_expr()) {
      assert(right_expr->column_expr().info().data_type() !=
             proto::schema::DataType::Array);
      assert(right_expr->column_expr().info().nested_path_size() == 0);
    }

    if (left_expr_with_type.dtype == proto::schema::DataType::Array) {
      if (right_expr_with_type.dtype == proto::schema::DataType::Array)
        assert(canArithmeticDtype(getArrayElementType(left_expr),
                                  getArrayElementType(right_expr)));
      else if (arithmeticDtype(left_expr_with_type.dtype))
        assert(canArithmeticDtype(getArrayElementType(left_expr),
                                  right_expr_with_type.dtype));
      else
        assert(false);
    }

    if (right_expr_with_type.dtype == proto::schema::DataType::Array) {
      if (arithmeticDtype(left_expr_with_type.dtype))
        assert(canArithmeticDtype(left_expr_with_type.dtype,
                                  getArrayElementType(right_expr)));
      else
        assert(false);
    }

    if (arithmeticDtype(left_expr_with_type.dtype) &&
        arithmeticDtype(right_expr_with_type.dtype)) {
      assert(canArithmeticDtype(left_expr_with_type.dtype,
                                right_expr_with_type.dtype));
    } else {
      assert(false);
    }

    switch (ctx->op->getType()) {
    case PlanParser::MUL:
      return ExprWithDtype(
          createBinArithExpr<proto::plan::ArithOpType::Mul>(
              left_expr, right_expr, this->arena.get()),
          calDataType(&left_expr_with_type, &right_expr_with_type), false);
    case PlanParser::DIV:
      return ExprWithDtype(
          createBinArithExpr<proto::plan::ArithOpType::Div>(
              left_expr, right_expr, this->arena.get()),
          calDataType(&left_expr_with_type, &right_expr_with_type), false);
    case PlanParser::MOD:
      return ExprWithDtype(
          createBinArithExpr<proto::plan::ArithOpType::Mod>(
              left_expr, right_expr, this->arena.get()),
          calDataType(&left_expr_with_type, &right_expr_with_type), false);

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
    auto expr =
        google::protobuf::Arena::CreateMessage<proto::plan::Expr>(arena.get());
    auto col_expr =
        google::protobuf::Arena::CreateMessage<proto::plan::ColumnExpr>(
            arena.get());
    auto info = google::protobuf::Arena::CreateMessage<proto::plan::ColumnInfo>(
        arena.get());
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
    auto child_expr_with_type =
        std::any_cast<ExprWithDtype>(ctx->expr()->accept(this));
    auto child_expr = child_expr_with_type.expr;
    assert(child_expr);
    auto info = child_expr->column_expr().info();
    assert(!(info.data_type() == proto::schema::DataType::JSON &&
             info.nested_path_size() == 0));
    assert((child_expr_with_type.dtype == proto::schema::DataType::String ||
            child_expr_with_type.dtype == proto::schema::DataType::JSON) ||
           (child_expr_with_type.dtype == proto::schema::DataType::Array &&
            info.element_type() == proto::schema::DataType::String));

    auto str = ctx->StringLiteral()->getText();
    auto pattern = convertEscapeSingle(str);

    auto res = translatePatternMatch(pattern);

    auto expr =
        google::protobuf::Arena::CreateMessage<proto::plan::Expr>(arena.get());
    auto unaryrange_expr =
        google::protobuf::Arena::CreateMessage<proto::plan::UnaryRangeExpr>(
            arena.get());
    unaryrange_expr->set_op(res.first);
    unaryrange_expr->set_allocated_column_info(
        new proto::plan::ColumnInfo(info));
    expr->set_allocated_unary_range_expr(unaryrange_expr);
    return ExprWithDtype(expr, proto::schema::DataType::Bool, false);
  }

  virtual std::any visitEquality(PlanParser::EqualityContext *ctx) override {
    auto left_expr_with_type =
        std::any_cast<ExprWithDtype>(ctx->expr()[0]->accept(this));
    auto right_expr_with_type =
        std::any_cast<ExprWithDtype>(ctx->expr()[1]->accept(this));

    auto left_value = extractValue(left_expr_with_type.expr);
    auto right_value = extractValue(right_expr_with_type.expr);

    if (left_value.has_value() && right_value.has_value()) {

#define PROCESS_EQALITY(left_type, right_type)                                 \
  if (left_value.type() == typeid(left_type) &&                                \
      right_value.type() == typeid(right_type)) {                              \
    switch (ctx->op->getType()) {                                              \
    case PlanParser::EQ:                                                       \
      return createValueExpr<bool>(std::any_cast<left_type>(left_value) ==     \
                                       std::any_cast<right_type>(right_value), \
                                   this->arena.get());                         \
    case PlanParser::NE:                                                       \
      return createValueExpr<bool>(std::any_cast<left_type>(left_value) !=     \
                                       std::any_cast<right_type>(right_value), \
                                   this->arena.get());                         \
    }                                                                          \
  }

      PROCESS_EQALITY(bool, bool);
      PROCESS_EQALITY(std::string, std::string);
      PROCESS_EQALITY(double, double);
      PROCESS_EQALITY(int32_t, int32_t);
      PROCESS_EQALITY(int64_t, int64_t);
      PROCESS_EQALITY(double, int32_t);
      PROCESS_EQALITY(int32_t, double);
      PROCESS_EQALITY(float, float);
      PROCESS_EQALITY(int32_t, float);
      PROCESS_EQALITY(float, int32_t);
      PROCESS_EQALITY(float, double);
      PROCESS_EQALITY(double, float);
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

    if (left_expr_with_type.expr->has_value_expr() &&
        !right_expr_with_type.expr->has_value_expr()) {
      ExprWithDtype left = toValueExpr(
          google::protobuf::Arena::CreateMessage<proto::plan::GenericValue>(
              this->arena.get(),
              left_expr_with_type.expr->value_expr().value()),
          this->arena.get());
      ExprWithDtype right = right_expr_with_type;

      return ExprWithDtype(HandleCompare(op, left, right, this->arena.get()),
                           proto::schema::DataType::Bool, false);
    }

    if (!left_expr_with_type.expr->has_value_expr() &&
        right_expr_with_type.expr->has_value_expr()) {

      ExprWithDtype left = left_expr_with_type;
      ExprWithDtype right = toValueExpr(
          google::protobuf::Arena::CreateMessage<proto::plan::GenericValue>(
              this->arena.get(),
              right_expr_with_type.expr->value_expr().value()));

      return ExprWithDtype(HandleCompare(op, left, right, this->arena.get()),
                           proto::schema::DataType::Bool, false);
    }

    if (left_expr_with_type.expr->has_value_expr() &&
        right_expr_with_type.expr->has_value_expr()) {
      ExprWithDtype left = toValueExpr(
          google::protobuf::Arena::CreateMessage<proto::plan::GenericValue>(
              this->arena.get(),
              left_expr_with_type.expr->value_expr().value()));
      ExprWithDtype right = toValueExpr(
          google::protobuf::Arena::CreateMessage<proto::plan::GenericValue>(
              this->arena.get(),
              right_expr_with_type.expr->value_expr().value()));

      return ExprWithDtype(HandleCompare(op, left, right, this->arena.get()),
                           proto::schema::DataType::Bool, false);
    }

    if (!left_expr_with_type.expr->has_value_expr() &&
        !right_expr_with_type.expr->has_value_expr()) {

      return ExprWithDtype(HandleCompare(op, left_expr_with_type,
                                         right_expr_with_type,
                                         this->arena.get()),
                           proto::schema::DataType::Bool, false);
    }
  }

  proto::plan::ColumnInfo *
  getChildColumnInfo(antlr4::tree::TerminalNode *identifier,
                     antlr4::tree::TerminalNode *child) {
    if (identifier) {

      auto text = identifier->getText();
      auto field = helper->GetFieldFromNameDefaultJSON(text);
      std::vector<std::string> nested_path;
      if (field.name() != text) {
        nested_path.push_back(text);
      }
      assert(!(field.data_type() == proto::schema::DataType::JSON &&
               nested_path.empty()));
      auto info =
          google::protobuf::Arena::CreateMessage<proto::plan::ColumnInfo>(
              this->arena.get());
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

    auto childtext = child->getText();
    std::string fieldname = childtext.substr(0, childtext.find("[", 0));

    std::vector<std::string> nestedpath;
    auto field = helper->GetFieldFromNameDefaultJSON(fieldname);
    assert(field.data_type() == proto::schema::DataType::JSON ||
           field.data_type() == proto::schema::DataType::Array);
    if (fieldname != field.name())
      nestedpath.push_back(fieldname);
    auto jsonkey = childtext.substr(fieldname.length(),
                                    childtext.length() - fieldname.length());
    auto ss = tokenize(jsonkey, "][");
    for (size_t i = 0; i < ss.size(); ++i) {
      std::string path_ = ss[i];
      if (path_[0] == '[')
        path_ = path_.substr(1, path_.length() - 1);
      if (path_[path_.length() - 1] == ']')
        path_ = ss[i].substr(0, path_.length() - 1);
      assert(path_ != "");

      if ((path_[0] == '\"' && path_[path_.length() - 1] == '\"') ||
          (path_[0] == '"' && path_[path_.length() - 1] == '"')) {
      }
    }

    return nullptr;
  }

  virtual std::any
  visitReverseRange(PlanParser::ReverseRangeContext *ctx) override {

    auto info = getChildColumnInfo(ctx->Identifier(), ctx->JSONIdentifier());
    assert(info != nullptr);
    assert(checkDirectComparisonBinaryField(info));
    auto lower = std::any_cast<ExprWithDtype>(ctx->expr()[1]->accept(this));
    auto upper = std::any_cast<ExprWithDtype>(ctx->expr()[0]->accept(this));

    {
      if (info->data_type() == proto::schema::DataType::Int8 ||
          info->data_type() == proto::schema::DataType::Int16 ||
          info->data_type() == proto::schema::DataType::Int32 ||
          info->data_type() == proto::schema::DataType::Int64) {
        auto a = extractValue<int64_t>(lower.expr);
        auto b = extractValue<int64_t>(upper.expr);
        if (a.has_value() && b.has_value()) {
          bool lowerinclusive = ctx->op1->getType() == PlanParser::GE;
          bool upperinclusive = ctx->op2->getType() == PlanParser::GE;
          auto expr = new proto::plan::Expr();
          auto binary_range_expr = new proto::plan::BinaryRangeExpr();
          auto lower_value = new proto::plan::GenericValue();
          auto upper_value = new proto::plan::GenericValue();
          lower_value->set_int64_val(a.value());
          upper_value->set_int64_val(b.value());
          binary_range_expr->set_lower_inclusive(lowerinclusive);
          binary_range_expr->set_upper_inclusive(upperinclusive);
          binary_range_expr->set_allocated_column_info(info);

          binary_range_expr->set_allocated_lower_value(lower_value);
          binary_range_expr->set_allocated_upper_value(upper_value);
          expr->set_allocated_binary_range_expr(binary_range_expr);
          return ExprWithDtype(expr, proto::schema::DataType::Bool, false);
        }
      }
    }

    {
      if (info->data_type() == proto::schema::DataType::Int8 ||
          info->data_type() == proto::schema::DataType::Int16 ||
          info->data_type() == proto::schema::DataType::Int32 ||
          info->data_type() == proto::schema::DataType::Int64) {
        auto a = extractValue<int32_t>(lower.expr);
        auto b = extractValue<int32_t>(upper.expr);
        if (a.has_value() && b.has_value()) {
          bool lowerinclusive = ctx->op1->getType() == PlanParser::GE;
          bool upperinclusive = ctx->op2->getType() == PlanParser::GE;
          auto expr = new proto::plan::Expr();
          auto binary_range_expr = new proto::plan::BinaryRangeExpr();
          auto lower_value = new proto::plan::GenericValue();
          auto upper_value = new proto::plan::GenericValue();
          lower_value->set_int64_val(a.value());
          upper_value->set_int64_val(b.value());
          binary_range_expr->set_lower_inclusive(lowerinclusive);
          binary_range_expr->set_upper_inclusive(upperinclusive);
          binary_range_expr->set_allocated_column_info(info);

          binary_range_expr->set_allocated_lower_value(lower_value);
          binary_range_expr->set_allocated_upper_value(upper_value);
          expr->set_allocated_binary_range_expr(binary_range_expr);
          return ExprWithDtype(expr, proto::schema::DataType::Bool, false);
        }
      }
    }

    {
      if (info->data_type() == proto::schema::DataType::Int8 ||
          info->data_type() == proto::schema::DataType::Int16 ||
          info->data_type() == proto::schema::DataType::Int32 ||
          info->data_type() == proto::schema::DataType::Int64) {
        auto a = extractValue<int8_t>(lower.expr);
        auto b = extractValue<int8_t>(upper.expr);
        if (a.has_value() && b.has_value()) {
          bool lowerinclusive = ctx->op1->getType() == PlanParser::GE;
          bool upperinclusive = ctx->op2->getType() == PlanParser::GE;
          auto expr = new proto::plan::Expr();
          auto binary_range_expr = new proto::plan::BinaryRangeExpr();
          auto lower_value = new proto::plan::GenericValue();
          auto upper_value = new proto::plan::GenericValue();
          lower_value->set_int64_val(a.value());
          upper_value->set_int64_val(b.value());
          binary_range_expr->set_lower_inclusive(lowerinclusive);
          binary_range_expr->set_upper_inclusive(upperinclusive);
          binary_range_expr->set_allocated_column_info(info);

          binary_range_expr->set_allocated_lower_value(lower_value);
          binary_range_expr->set_allocated_upper_value(upper_value);
          expr->set_allocated_binary_range_expr(binary_range_expr);
          return ExprWithDtype(expr, proto::schema::DataType::Bool, false);
        }
      }
    }
    {
      if (info->data_type() == proto::schema::DataType::Int8 ||
          info->data_type() == proto::schema::DataType::Int16 ||
          info->data_type() == proto::schema::DataType::Int32 ||
          info->data_type() == proto::schema::DataType::Int64) {
        auto a = extractValue<int16_t>(lower.expr);
        auto b = extractValue<int16_t>(upper.expr);
        if (a.has_value() && b.has_value()) {
          bool lowerinclusive = ctx->op1->getType() == PlanParser::GE;
          bool upperinclusive = ctx->op2->getType() == PlanParser::GE;
          auto expr = new proto::plan::Expr();
          auto binary_range_expr = new proto::plan::BinaryRangeExpr();
          auto lower_value = new proto::plan::GenericValue();
          auto upper_value = new proto::plan::GenericValue();
          lower_value->set_int64_val(a.value());
          upper_value->set_int64_val(b.value());
          binary_range_expr->set_lower_inclusive(lowerinclusive);
          binary_range_expr->set_upper_inclusive(upperinclusive);
          binary_range_expr->set_allocated_column_info(info);

          binary_range_expr->set_allocated_lower_value(lower_value);
          binary_range_expr->set_allocated_upper_value(upper_value);
          expr->set_allocated_binary_range_expr(binary_range_expr);
          return ExprWithDtype(expr, proto::schema::DataType::Bool, false);
        }
      }
    }

    {
      if (info->data_type() == proto::schema::DataType::Double ||
          info->data_type() == proto::schema::DataType::Float) {
        auto a = extractValue<double>(lower.expr);
        auto b = extractValue<double>(upper.expr);
        if (a.has_value() && b.has_value()) {
          bool lowerinclusive = ctx->op1->getType() == PlanParser::GE;
          bool upperinclusive = ctx->op2->getType() == PlanParser::GE;
          auto expr = new proto::plan::Expr();
          auto binary_range_expr = new proto::plan::BinaryRangeExpr();
          auto lower_value = new proto::plan::GenericValue();
          auto upper_value = new proto::plan::GenericValue();
          lower_value->set_float_val(a.value());
          upper_value->set_float_val(b.value());
          binary_range_expr->set_lower_inclusive(lowerinclusive);
          binary_range_expr->set_upper_inclusive(upperinclusive);
          binary_range_expr->set_allocated_column_info(info);

          binary_range_expr->set_allocated_lower_value(lower_value);
          binary_range_expr->set_allocated_upper_value(upper_value);
          expr->set_allocated_binary_range_expr(binary_range_expr);
          return ExprWithDtype(expr, proto::schema::DataType::Bool, false);
        }
      }
    }
    {
      if (info->data_type() == proto::schema::DataType::Double ||
          info->data_type() == proto::schema::DataType::Float) {
        auto a = extractValue<float>(lower.expr);
        auto b = extractValue<float>(upper.expr);
        if (a.has_value() && b.has_value()) {
          bool lowerinclusive = ctx->op1->getType() == PlanParser::GE;
          bool upperinclusive = ctx->op2->getType() == PlanParser::GE;
          auto expr = new proto::plan::Expr();
          auto binary_range_expr = new proto::plan::BinaryRangeExpr();
          auto lower_value = new proto::plan::GenericValue();
          auto upper_value = new proto::plan::GenericValue();
          lower_value->set_float_val(double(a.value()));
          upper_value->set_float_val(double(b.value()));
          binary_range_expr->set_lower_inclusive(lowerinclusive);
          binary_range_expr->set_upper_inclusive(upperinclusive);
          binary_range_expr->set_allocated_column_info(info);

          binary_range_expr->set_allocated_lower_value(lower_value);
          binary_range_expr->set_allocated_upper_value(upper_value);
          expr->set_allocated_binary_range_expr(binary_range_expr);
          return ExprWithDtype(expr, proto::schema::DataType::Bool, false);
        }
      }
    }
    return nullptr;
  }

  virtual std::any visitAddSub(PlanParser::AddSubContext *ctx) override {
    auto a_expr_with_type =
        std::any_cast<ExprWithDtype>(ctx->expr()[0]->accept(this));
    auto b_expr_with_type =
        std::any_cast<ExprWithDtype>(ctx->expr()[1]->accept(this));
    auto a = a_expr_with_type.expr;
    auto b = b_expr_with_type.expr;
    if (extractValue<double>(a).has_value() &&
        extractValue<double>(b).has_value()) {
      switch (ctx->op->getType()) {
      case PlanParser::ADD:
        return ExprWithDtype(
            createValueExpr<double>(extractValue<double>(a).value() +
                                    extractValue<double>(b).value()),
            proto::schema::DataType::Double, false);
      case PlanParser::SUB:
        return ExprWithDtype(
            createValueExpr<double>(extractValue<double>(a).value() -
                                    extractValue<double>(b).value()),
            proto::schema::DataType::Double, false);
      default:
        assert(false);
      }
    }

    if (extractValue<int64_t>(a).has_value() &&
        extractValue<int64_t>(b).has_value()) {
      switch (ctx->op->getType()) {
      case PlanParser::ADD:
        return createValueExpr<int64_t>(extractValue<int64_t>(a).value() +
                                        extractValue<int64_t>(b).value());
      case PlanParser::SUB:
        return createValueExpr<int64_t>(extractValue<int64_t>(a).value() -
                                        extractValue<int64_t>(b).value());
      default:
        assert(false);
      }
    }

    if (extractValue<double>(a).has_value() &&
        extractValue<int64_t>(b).has_value()) {
      switch (ctx->op->getType()) {
      case PlanParser::ADD:
        return createValueExpr<double>(
            extractValue<double>(a).value() +
            double(extractValue<int64_t>(b).value()));
      case PlanParser::SUB:
        return createValueExpr<double>(
            extractValue<double>(a).value() -
            double(extractValue<int64_t>(b).value()));
      default:
        assert(false);
      }
    }

    if (extractValue<int64_t>(a).has_value() &&
        extractValue<double>(b).has_value()) {
      switch (ctx->op->getType()) {
      case PlanParser::ADD:
        return double(extractValue<int64_t>(a).value()) +
               extractValue<double>(b).value();
      case PlanParser::SUB:
        return double(extractValue<int64_t>(a).value()) -
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
    case PlanParser::ADD:
      return ExprWithDtype(
          createBinArithExpr<proto::plan::ArithOpType::Add>(a, b),
          calDataType(&a_expr_with_type, &b_expr_with_type), false);
    case PlanParser::DIV:
      return ExprWithDtype(
          createBinArithExpr<proto::plan::ArithOpType::Sub>(a, b),
          calDataType(&a_expr_with_type, &b_expr_with_type), false);

    default:
      assert(false);
    }
  }

  virtual std::any
  visitRelational(PlanParser::RelationalContext *ctx) override {
    auto a_expr_with_type =
        std::any_cast<ExprWithDtype>(ctx->expr()[0]->accept(this));
    auto b_expr_with_type =
        std::any_cast<ExprWithDtype>(ctx->expr()[1]->accept(this));
    auto a = a_expr_with_type.expr;
    auto b = b_expr_with_type.expr;
    if (extractValue<double>(a).has_value() &&
        extractValue<double>(b).has_value()) {
      switch (ctx->op->getType()) {
      case PlanParser::LT:
        return ExprWithDtype(
            createValueExpr<bool>(extractValue<double>(a).value() <
                                  extractValue<double>(b).value()),
            proto::schema::DataType::Bool, false);
      case PlanParser::LE:
        return ExprWithDtype(
            createValueExpr<bool>(extractValue<double>(a).value() <=
                                  extractValue<double>(b).value()),
            proto::schema::DataType::Bool, false);
      case PlanParser::GT:
        return ExprWithDtype(
            createValueExpr<bool>(extractValue<double>(a).value() >
                                  extractValue<double>(b).value()),
            proto::schema::DataType::Bool, false);
      case PlanParser::GE:
        return ExprWithDtype(
            createValueExpr<bool>(extractValue<double>(a).value() >=
                                  extractValue<double>(b).value()),
            proto::schema::DataType::Bool, false);

      default:
        assert(false);
      }
    }

    if (extractValue<int64_t>(a).has_value() &&
        extractValue<int64_t>(b).has_value()) {
      switch (ctx->op->getType()) {

      case PlanParser::LT:
        return ExprWithDtype(
            createValueExpr<bool>(extractValue<int64_t>(a).value() <
                                  extractValue<int64_t>(b).value()),
            proto::schema::DataType::Bool, false);
      case PlanParser::LE:
        return ExprWithDtype(
            createValueExpr<bool>(extractValue<int64_t>(a).value() <=
                                  extractValue<int64_t>(b).value()),
            proto::schema::DataType::Bool, false);
      case PlanParser::GT:
        return ExprWithDtype(
            createValueExpr<bool>(extractValue<int64_t>(a).value() >
                                  extractValue<int64_t>(b).value()),
            proto::schema::DataType::Bool, false);
      case PlanParser::GE:
        return ExprWithDtype(
            createValueExpr<bool>(extractValue<int64_t>(a).value() >=
                                  extractValue<int64_t>(b).value()),
            proto::schema::DataType::Bool, false);

      default:
        assert(false);
      }
    }

    if (extractValue<double>(a).has_value() &&
        extractValue<int64_t>(b).has_value()) {
      switch (ctx->op->getType()) {

      case PlanParser::LT:
        return ExprWithDtype(
            createValueExpr<bool>(extractValue<double>(a).value() <
                                  double(extractValue<int64_t>(b).value())),
            proto::schema::DataType::Bool, false);
      case PlanParser::LE:
        return ExprWithDtype(
            createValueExpr<bool>(extractValue<double>(a).value() <=
                                  double(extractValue<int64_t>(b).value())),
            proto::schema::DataType::Bool, false);
      case PlanParser::GT:
        return ExprWithDtype(
            createValueExpr<bool>(extractValue<double>(a).value() >
                                  double(extractValue<int64_t>(b).value())),
            proto::schema::DataType::Bool, false);
      case PlanParser::GE:
        return ExprWithDtype(
            createValueExpr<bool>(extractValue<double>(a).value() >=
                                  double(extractValue<int64_t>(b).value())),
            proto::schema::DataType::Bool, false);

      default:
        assert(false);
      }
    }

    if (extractValue<int64_t>(a).has_value() &&
        extractValue<double>(b).has_value()) {
      switch (ctx->op->getType()) {

      case PlanParser::LT:
        return ExprWithDtype(
            createValueExpr<bool>(double(extractValue<int64_t>(a).value()) <
                                  extractValue<double>(b).value()),
            proto::schema::DataType::Bool, false);
      case PlanParser::LE:
        return ExprWithDtype(
            createValueExpr<bool>(double(extractValue<int64_t>(a).value()) <=
                                  extractValue<double>(b).value()),
            proto::schema::DataType::Bool, false);
      case PlanParser::GT:
        return ExprWithDtype(
            createValueExpr<bool>(double(extractValue<int64_t>(a).value()) >
                                  extractValue<double>(b).value()),
            proto::schema::DataType::Bool, false);
      case PlanParser::GE:
        return ExprWithDtype(
            createValueExpr<bool>(double(extractValue<int64_t>(a).value()) >=
                                  double(extractValue<double>(b).value())),
            proto::schema::DataType::Bool, false);

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

    if (a->has_value_expr() && !b->has_value_expr()) {
      ExprWithDtype left =
          toValueExpr(new proto::plan::GenericValue(a->value_expr().value()));
      ExprWithDtype right = b_expr_with_type;

      return ExprWithDtype(HandleCompare(op, left, right),
                           proto::schema::DataType::Bool, false);
    }

    if (!a->has_value_expr() && b->has_value_expr()) {

      ExprWithDtype left = a_expr_with_type;
      ExprWithDtype right =
          toValueExpr(new proto::plan::GenericValue(b->value_expr().value()));

      return ExprWithDtype(HandleCompare(op, left, right),
                           proto::schema::DataType::Bool, false);
    }

    if (a->has_value_expr() && b->has_value_expr()) {

      ExprWithDtype left =
          toValueExpr(new proto::plan::GenericValue(a->value_expr().value()));
      ExprWithDtype right =
          toValueExpr(new proto::plan::GenericValue(b->value_expr().value()));

      return ExprWithDtype(HandleCompare(op, left, right),
                           proto::schema::DataType::Bool, false);
    }

    if (!a->has_value_expr() && !b->has_value_expr()) {

      return ExprWithDtype(
          HandleCompare(op, a_expr_with_type, b_expr_with_type),
          proto::schema::DataType::Bool, false);
    }
    return nullptr;
  }

  virtual std::any
  visitArrayLength(PlanParser::ArrayLengthContext *ctx) override {

    auto info = getChildColumnInfo(ctx->Identifier(), ctx->JSONIdentifier());
    assert(info);
    assert(info->data_type() == proto::schema::Array ||
           info->data_type() == proto::schema::JSON);
    auto expr = new proto::plan::Expr();
    auto bin_arith_expr = new proto::plan::BinaryArithExpr();
    auto column_expr = new proto::plan::ColumnExpr();
    column_expr->set_allocated_info(info);
    auto left_expr = new proto::plan::Expr();
    left_expr->set_allocated_column_expr(column_expr);
    bin_arith_expr->set_allocated_left(left_expr);
    bin_arith_expr->set_op(proto::plan::ArithOpType::ArrayLength);
    expr->set_allocated_binary_arith_expr(bin_arith_expr);
    return ExprWithDtype(expr, proto::schema::DataType::Int64, false);
  }

  virtual std::any visitTerm(PlanParser::TermContext *ctx) override {
    auto first = std::any_cast<ExprWithDtype>(ctx->expr()[0]->accept(this));
    auto info = first.expr->unsafe_arena_release_column_expr()
                    ->unsafe_arena_release_info();

    auto term_expr =
        google::protobuf::Arena::CreateMessage<proto::plan::TermExpr>(
            arena.get());
    for (size_t i = 1; i < ctx->expr().size(); ++i) {
      auto elem = ctx->expr()[i];
      auto expr_ = std::any_cast<ExprWithDtype>(elem->accept(this)).expr;
      auto v =
          google::protobuf::Arena::CreateMessage<proto::plan::GenericValue>(
              arena.get());
      if (extractValue<int8_t>(expr_).has_value()) {
        v->set_int64_val(int64_t(extractValue<int8_t>(expr_).value()));
        term_expr->mutable_values()->UnsafeArenaAddAllocated(v);
        continue;
      }
      if (extractValue<int16_t>(expr_).has_value()) {
        v->set_int64_val(int64_t(extractValue<int16_t>(expr_).value()));
        term_expr->mutable_values()->UnsafeArenaAddAllocated(v);

        continue;
      }
      if (extractValue<int32_t>(expr_).has_value()) {
        v->set_int64_val(int64_t(extractValue<int32_t>(expr_).value()));
        term_expr->mutable_values()->UnsafeArenaAddAllocated(v);

        continue;
      }
      if (extractValue<int64_t>(expr_).has_value()) {
        v->set_int64_val(extractValue<int64_t>(expr_).value());
        term_expr->mutable_values()->UnsafeArenaAddAllocated(v);

        continue;
      }
      if (extractValue<float>(expr_).has_value()) {
        v->set_float_val(double(extractValue<float>(expr_).value()));
        term_expr->mutable_values()->UnsafeArenaAddAllocated(v);

        continue;
      }
      if (extractValue<double>(expr_).has_value()) {
        v->set_int64_val(double(extractValue<double>(expr_).value()));
        term_expr->mutable_values()->UnsafeArenaAddAllocated(v);

        continue;
      }
      assert(false);
    }
    auto expr =
        google::protobuf::Arena::CreateMessage<proto::plan::Expr>(arena.get());

    term_expr->unsafe_arena_set_allocated_column_info(info);
    expr->unsafe_arena_set_allocated_term_expr(term_expr);
    if (ctx->op->getType() == PlanParser::NIN) {
      auto root_expr =
          google::protobuf::Arena::CreateMessage<proto::plan::Expr>(
              arena.get());
      auto unary_expr =
          google::protobuf::Arena::CreateMessage<proto::plan::UnaryExpr>(
              arena.get());
      unary_expr->set_op(proto::plan::UnaryExpr_UnaryOp_Not);
      unary_expr->set_allocated_child(expr);
      return ExprWithDtype(root_expr, proto::schema::DataType::Bool, false);
    }
    std::cout << expr->DebugString() << std::endl;
    return ExprWithDtype(expr, proto::schema::DataType::Bool, false);
  }

  virtual std::any
  visitJSONContains(PlanParser::JSONContainsContext *ctx) override {
    auto field = std::any_cast<ExprWithDtype>(ctx->expr()[0]->accept(this));
    auto info = field.expr->column_expr().info();
    assert(info.data_type() == proto::schema::DataType::Array ||
           info.data_type() == proto::schema::DataType::JSON);
    auto elem = std::any_cast<ExprWithDtype>(ctx->expr()[1]->accept(this));
    if (info.data_type() == proto::schema::DataType::Array) {
      proto::plan::GenericValue expr =
          proto::plan::GenericValue(elem.expr->value_expr().value());
      assert(canBeCompared(field, toValueExpr(&expr)));
    }

    auto expr = new proto::plan::Expr();
    auto json_contain_expr = new proto::plan::JSONContainsExpr();
    auto value = json_contain_expr->add_elements();
    value->set_allocated_array_val(
        new proto::plan::Array(elem.expr->value_expr().value().array_val()));
    json_contain_expr->set_elements_same_type(true);
    json_contain_expr->set_allocated_column_info(
        new proto::plan::ColumnInfo(info));
    json_contain_expr->set_op(proto::plan::JSONContainsExpr_JSONOp_Contains);
    expr->set_allocated_json_contains_expr(json_contain_expr);
    return ExprWithDtype(expr, proto::schema::Bool, false);
  }

  virtual std::any visitRange(PlanParser::RangeContext *ctx) override {

    auto info = getChildColumnInfo(ctx->Identifier(), ctx->JSONIdentifier());
    assert(info != nullptr);
    assert(checkDirectComparisonBinaryField(info));
    auto lower = std::any_cast<ExprWithDtype>(ctx->expr()[0]->accept(this));
    auto upper = std::any_cast<ExprWithDtype>(ctx->expr()[1]->accept(this));

    {
      if (info->data_type() == proto::schema::DataType::Int8 ||
          info->data_type() == proto::schema::DataType::Int16 ||
          info->data_type() == proto::schema::DataType::Int32 ||
          info->data_type() == proto::schema::DataType::Int64) {
        auto a = extractValue<int64_t>(lower.expr);
        auto b = extractValue<int64_t>(upper.expr);
        if (a.has_value() && b.has_value()) {
          bool lowerinclusive = ctx->op1->getType() == PlanParser::LE;
          bool upperinclusive = ctx->op2->getType() == PlanParser::LE;
          auto expr = new proto::plan::Expr();
          auto binary_range_expr = new proto::plan::BinaryRangeExpr();
          auto lower_value = new proto::plan::GenericValue();
          auto upper_value = new proto::plan::GenericValue();
          lower_value->set_int64_val(a.value());
          upper_value->set_int64_val(b.value());
          binary_range_expr->set_lower_inclusive(lowerinclusive);
          binary_range_expr->set_upper_inclusive(upperinclusive);
          binary_range_expr->set_allocated_column_info(info);

          binary_range_expr->set_allocated_lower_value(lower_value);
          binary_range_expr->set_allocated_upper_value(upper_value);
          expr->set_allocated_binary_range_expr(binary_range_expr);
          return ExprWithDtype(expr, proto::schema::DataType::Bool, false);
        }
      }
    }

    {
      if (info->data_type() == proto::schema::DataType::Int8 ||
          info->data_type() == proto::schema::DataType::Int16 ||
          info->data_type() == proto::schema::DataType::Int32 ||
          info->data_type() == proto::schema::DataType::Int64) {
        auto a = extractValue<int32_t>(lower.expr);
        auto b = extractValue<int32_t>(upper.expr);
        if (a.has_value() && b.has_value()) {
          bool lowerinclusive = ctx->op1->getType() == PlanParser::LE;
          bool upperinclusive = ctx->op2->getType() == PlanParser::LE;
          auto expr = new proto::plan::Expr();
          auto binary_range_expr = new proto::plan::BinaryRangeExpr();
          auto lower_value = new proto::plan::GenericValue();
          auto upper_value = new proto::plan::GenericValue();
          lower_value->set_int64_val(a.value());
          upper_value->set_int64_val(b.value());
          binary_range_expr->set_lower_inclusive(lowerinclusive);
          binary_range_expr->set_upper_inclusive(upperinclusive);
          binary_range_expr->set_allocated_column_info(info);

          binary_range_expr->set_allocated_lower_value(lower_value);
          binary_range_expr->set_allocated_upper_value(upper_value);
          expr->set_allocated_binary_range_expr(binary_range_expr);
          return ExprWithDtype(expr, proto::schema::DataType::Bool, false);
        }
      }
    }

    {
      if (info->data_type() == proto::schema::DataType::Int8 ||
          info->data_type() == proto::schema::DataType::Int16 ||
          info->data_type() == proto::schema::DataType::Int32 ||
          info->data_type() == proto::schema::DataType::Int64) {
        auto a = extractValue<int8_t>(lower.expr);
        auto b = extractValue<int8_t>(upper.expr);
        if (a.has_value() && b.has_value()) {
          bool lowerinclusive = ctx->op1->getType() == PlanParser::LE;
          bool upperinclusive = ctx->op2->getType() == PlanParser::LE;
          auto expr = new proto::plan::Expr();
          auto binary_range_expr = new proto::plan::BinaryRangeExpr();
          auto lower_value = new proto::plan::GenericValue();
          auto upper_value = new proto::plan::GenericValue();
          lower_value->set_int64_val(a.value());
          upper_value->set_int64_val(b.value());
          binary_range_expr->set_lower_inclusive(lowerinclusive);
          binary_range_expr->set_upper_inclusive(upperinclusive);
          binary_range_expr->set_allocated_column_info(info);

          binary_range_expr->set_allocated_lower_value(lower_value);
          binary_range_expr->set_allocated_upper_value(upper_value);
          expr->set_allocated_binary_range_expr(binary_range_expr);
          return ExprWithDtype(expr, proto::schema::DataType::Bool, false);
        }
      }
    }
    {
      if (info->data_type() == proto::schema::DataType::Int8 ||
          info->data_type() == proto::schema::DataType::Int16 ||
          info->data_type() == proto::schema::DataType::Int32 ||
          info->data_type() == proto::schema::DataType::Int64) {
        auto a = extractValue<int16_t>(lower.expr);
        auto b = extractValue<int16_t>(upper.expr);
        if (a.has_value() && b.has_value()) {
          bool lowerinclusive = ctx->op1->getType() == PlanParser::LE;
          bool upperinclusive = ctx->op2->getType() == PlanParser::LE;
          auto expr = new proto::plan::Expr();
          auto binary_range_expr = new proto::plan::BinaryRangeExpr();
          auto lower_value = new proto::plan::GenericValue();
          auto upper_value = new proto::plan::GenericValue();
          lower_value->set_int64_val(a.value());
          upper_value->set_int64_val(b.value());
          binary_range_expr->set_lower_inclusive(lowerinclusive);
          binary_range_expr->set_upper_inclusive(upperinclusive);
          binary_range_expr->set_allocated_column_info(info);

          binary_range_expr->set_allocated_lower_value(lower_value);
          binary_range_expr->set_allocated_upper_value(upper_value);
          expr->set_allocated_binary_range_expr(binary_range_expr);
          return ExprWithDtype(expr, proto::schema::DataType::Bool, false);
        }
      }
    }

    {
      if (info->data_type() == proto::schema::DataType::Double ||
          info->data_type() == proto::schema::DataType::Float) {
        auto a = extractValue<double>(lower.expr);
        auto b = extractValue<double>(upper.expr);
        if (a.has_value() && b.has_value()) {
          bool lowerinclusive = ctx->op1->getType() == PlanParser::LE;
          bool upperinclusive = ctx->op2->getType() == PlanParser::LE;
          auto expr = new proto::plan::Expr();
          auto binary_range_expr = new proto::plan::BinaryRangeExpr();
          auto lower_value = new proto::plan::GenericValue();
          auto upper_value = new proto::plan::GenericValue();
          lower_value->set_float_val(a.value());
          upper_value->set_float_val(b.value());
          binary_range_expr->set_lower_inclusive(lowerinclusive);
          binary_range_expr->set_upper_inclusive(upperinclusive);
          binary_range_expr->set_allocated_column_info(info);

          binary_range_expr->set_allocated_lower_value(lower_value);
          binary_range_expr->set_allocated_upper_value(upper_value);
          expr->set_allocated_binary_range_expr(binary_range_expr);
          return ExprWithDtype(expr, proto::schema::DataType::Bool, false);
        }
      }
    }
    {
      if (info->data_type() == proto::schema::DataType::Double ||
          info->data_type() == proto::schema::DataType::Float) {
        auto a = extractValue<float>(lower.expr);
        auto b = extractValue<float>(upper.expr);
        if (a.has_value() && b.has_value()) {
          bool lowerinclusive = ctx->op1->getType() == PlanParser::LE;
          bool upperinclusive = ctx->op2->getType() == PlanParser::LE;
          auto expr = new proto::plan::Expr();
          auto binary_range_expr = new proto::plan::BinaryRangeExpr();
          auto lower_value = new proto::plan::GenericValue();
          auto upper_value = new proto::plan::GenericValue();
          lower_value->set_float_val(double(a.value()));
          upper_value->set_float_val(double(b.value()));
          binary_range_expr->set_lower_inclusive(lowerinclusive);
          binary_range_expr->set_upper_inclusive(upperinclusive);
          binary_range_expr->set_allocated_column_info(info);

          binary_range_expr->set_allocated_lower_value(lower_value);
          binary_range_expr->set_allocated_upper_value(upper_value);
          expr->set_allocated_binary_range_expr(binary_range_expr);
          return ExprWithDtype(expr, proto::schema::DataType::Bool, false);
        }
      }
    }
    return nullptr;
  }

  virtual std::any visitUnary(PlanParser::UnaryContext *ctx) override {

    auto a = std::any_cast<ExprWithDtype>(ctx->expr()->accept(this));
    if (extractValue<double>(a.expr).has_value()) {
      switch (ctx->op->getType()) {
      case PlanParser::ADD:
        return a;
      case PlanParser::SUB:
        return ExprWithDtype(
            createValueExpr<double>(-extractValue<double>(a.expr).value()),
            proto::schema::DataType::Float, false);
      case PlanParser::NOT:
        return ExprWithDtype(
            createValueExpr<bool>(!extractValue<double>(a.expr).value()),
            proto::schema::DataType::Bool, false);

      default:
        assert(false);
      }
    }

    if (extractValue<float>(a.expr).has_value()) {
      switch (ctx->op->getType()) {
      case PlanParser::ADD:
        return a;
      case PlanParser::SUB:
        return ExprWithDtype(
            createValueExpr<float>(-extractValue<float>(a.expr).value()),
            proto::schema::DataType::Float, false);
      case PlanParser::NOT:
        return ExprWithDtype(
            createValueExpr<bool>(!extractValue<float>(a.expr).value()),
            proto::schema::DataType::Bool, false);

      default:
        assert(false);
      }
    }
    if (extractValue<int8_t>(a.expr).has_value()) {
      switch (ctx->op->getType()) {
      case PlanParser::ADD:
        return a;
      case PlanParser::SUB:
        return ExprWithDtype(
            createValueExpr<int8_t>(-extractValue<int8_t>(a.expr).value()),
            proto::schema::DataType::Float, false);
      case PlanParser::NOT:
        return ExprWithDtype(
            createValueExpr<bool>(!extractValue<int8_t>(a.expr).value()),
            proto::schema::DataType::Bool, false);

      default:
        assert(false);
      }
    }

    if (extractValue<int16_t>(a.expr).has_value()) {
      switch (ctx->op->getType()) {
      case PlanParser::ADD:
        return a;
      case PlanParser::SUB:
        return ExprWithDtype(
            createValueExpr<int16_t>(-extractValue<int16_t>(a.expr).value()),
            proto::schema::DataType::Float, false);
      case PlanParser::NOT:
        return ExprWithDtype(
            createValueExpr<bool>(!extractValue<int16_t>(a.expr).value()),
            proto::schema::DataType::Bool, false);

      default:
        assert(false);
      }
    }

    if (extractValue<int32_t>(a.expr).has_value()) {
      switch (ctx->op->getType()) {
      case PlanParser::ADD:
        return a;
      case PlanParser::SUB:
        return ExprWithDtype(
            createValueExpr<int32_t>(-extractValue<int32_t>(a.expr).value()),
            proto::schema::DataType::Float, false);
      case PlanParser::NOT:
        return ExprWithDtype(
            createValueExpr<bool>(!extractValue<int32_t>(a.expr).value()),
            proto::schema::DataType::Bool, false);

      default:
        assert(false);
      }
    }
    if (extractValue<int64_t>(a.expr).has_value()) {
      switch (ctx->op->getType()) {
      case PlanParser::ADD:
        return a;
      case PlanParser::SUB:
        return ExprWithDtype(
            createValueExpr<int64_t>(-extractValue<int64_t>(a.expr).value()),
            proto::schema::DataType::Float, false);
      case PlanParser::NOT:
        return ExprWithDtype(
            createValueExpr<bool>(!extractValue<int64_t>(a.expr).value()),
            proto::schema::DataType::Bool, false);

      default:
        assert(false);
      }
    }

    assert(checkDirectComparisonBinaryField(
        new proto::plan::ColumnInfo(a.expr->column_expr().info())));
    switch (ctx->op->getType()) {
    case PlanParser::ADD:
      return a;
    case PlanParser::NOT:
      assert(!a.dependent && a.dtype == proto::schema::DataType::Bool);
      auto expr = new proto::plan::Expr();
      auto unary_expr = new proto::plan::UnaryExpr();
      unary_expr->set_allocated_child(a.expr);
      unary_expr->set_op(proto::plan::UnaryExpr_UnaryOp_Not);
      return ExprWithDtype(expr, proto::schema::Bool, false);
    }
    return nullptr;
  }

  virtual std::any visitArray(PlanParser::ArrayContext *ctx) override {

    auto expr = new proto::plan::Expr();
    auto array_expr = new proto::plan::Array();
    auto dtype = proto::schema::DataType::None;
    auto is_same = true;
    for (auto &&elem : ctx->expr()) {
      auto expr_ = std::any_cast<ExprWithDtype>(elem->accept(this)).expr;
      auto v = array_expr->add_array();
      if (extractValue<int8_t>(expr_).has_value()) {
        v->set_int64_val(int64_t(extractValue<int8_t>(expr_).value()));

        if (dtype != proto::schema::DataType::None &&
            dtype != proto::schema::DataType::Int8) {
          is_same = false;
        }

        if (dtype == proto::schema::DataType::None) {
          dtype = proto::schema::DataType::Int8;
        }
        continue;
      }
      if (extractValue<int16_t>(expr_).has_value()) {
        v->set_int64_val(int64_t(extractValue<int16_t>(expr_).value()));
        if (dtype != proto::schema::DataType::None &&
            dtype != proto::schema::DataType::Int16) {
          is_same = false;
        }

        if (dtype == proto::schema::DataType::None) {
          dtype = proto::schema::DataType::Int16;
        }

        continue;
      }
      if (extractValue<int32_t>(expr_).has_value()) {
        v->set_int64_val(int64_t(extractValue<int32_t>(expr_).value()));
        if (dtype != proto::schema::DataType::None &&
            dtype != proto::schema::DataType::Int32) {
          is_same = false;
        }

        if (dtype == proto::schema::DataType::None) {
          dtype = proto::schema::DataType::Int32;
        }

        continue;
      }
      if (extractValue<int64_t>(expr_).has_value()) {
        v->set_int64_val(extractValue<int64_t>(expr_).value());
        if (dtype != proto::schema::DataType::None &&
            dtype != proto::schema::DataType::Int64) {
          is_same = false;
        }

        if (dtype == proto::schema::DataType::None) {
          dtype = proto::schema::DataType::Int64;
        }

        continue;
      }
      if (extractValue<float>(expr_).has_value()) {
        v->set_float_val(double(extractValue<float>(expr_).value()));
        if (dtype != proto::schema::DataType::None &&
            dtype != proto::schema::DataType::Float) {
          is_same = false;
        }

        if (dtype == proto::schema::DataType::None) {
          dtype = proto::schema::DataType::Float;
        }

        continue;
      }
      if (extractValue<double>(expr_).has_value()) {
        v->set_int64_val(double(extractValue<double>(expr_).value()));
        if (dtype != proto::schema::DataType::None &&
            dtype != proto::schema::DataType::Double) {
          is_same = false;
        }

        if (dtype == proto::schema::DataType::None) {
          dtype = proto::schema::DataType::Double;
        }

        continue;
      }
      assert(false);
    }

    auto value = new proto::plan::GenericValue();
    value->set_allocated_array_val(array_expr);
    auto value_expr = new proto::plan::ValueExpr();
    expr->set_allocated_value_expr(value_expr);
    return ExprWithDtype(expr, is_same ? dtype : proto::schema::DataType::None,
                         true);
  }

  virtual std::any
  visitJSONContainsAny(PlanParser::JSONContainsAnyContext *ctx) override {

    auto field = std::any_cast<ExprWithDtype>(ctx->expr()[0]->accept(this));
    auto info = field.expr->column_expr().info();
    assert(info.data_type() == proto::schema::DataType::Array ||
           info.data_type() == proto::schema::DataType::JSON);
    auto elem = std::any_cast<ExprWithDtype>(ctx->expr()[1]->accept(this));
    if (info.data_type() == proto::schema::DataType::Array) {
      proto::plan::GenericValue expr =
          proto::plan::GenericValue(elem.expr->value_expr().value());
      assert(canBeCompared(field, toValueExpr(&expr)));
    }

    auto expr = new proto::plan::Expr();
    auto json_contain_expr = new proto::plan::JSONContainsExpr();
    auto value = json_contain_expr->add_elements();
    value->set_allocated_array_val(
        new proto::plan::Array(elem.expr->value_expr().value().array_val()));
    json_contain_expr->set_elements_same_type(
        elem.expr->value_expr().value().array_val().same_type());
    json_contain_expr->set_allocated_column_info(
        new proto::plan::ColumnInfo(info));
    json_contain_expr->set_op(proto::plan::JSONContainsExpr_JSONOp_ContainsAny);
    expr->set_allocated_json_contains_expr(json_contain_expr);
    return ExprWithDtype(expr, proto::schema::Bool, false);
  }

  virtual std::any visitExists(PlanParser::ExistsContext *ctx) override {
    auto a = std::any_cast<ExprWithDtype>(ctx->expr());
    auto info = a.expr->column_expr().info();
    assert(info.data_type() == proto::schema::DataType::Array);
    auto expr = new proto::plan::Expr();
    auto col_expr = new proto::plan::ColumnExpr();
    col_expr->set_allocated_info(new proto::plan::ColumnInfo(info));
    expr->set_allocated_column_expr(col_expr);
    return ExprWithDtype(expr, proto::schema::DataType::Bool, false);
  }

  virtual std::any visitEmptyTerm(PlanParser::EmptyTermContext *ctx) override {

    auto first = std::any_cast<ExprWithDtype>(ctx->expr()->accept(this));
    auto info = first.expr->column_expr().info();

    auto expr = new proto::plan::Expr();
    auto col_expr = new proto::plan::ColumnExpr();
    auto term_expr = new proto::plan::TermExpr();
    expr->set_allocated_term_expr(term_expr);
    col_expr->set_allocated_info(new proto::plan::ColumnInfo(info));
    expr->set_allocated_column_expr(col_expr);
    expr->set_allocated_term_expr(term_expr);
    return ExprWithDtype(expr, proto::schema::DataType::Bool, false);
  }

  PlanCCVisitor(SchemaHelper *const helper)
      : helper(helper), arena(std::make_shared<google::protobuf::Arena>()) {}

private:
  SchemaHelper *helper;
  std::shared_ptr<google::protobuf::Arena> arena;
};

} // namespace milvus

int main(int, const char *argv[]) {

  std::string exprstr(argv[1]);
  antlr4::ANTLRInputStream input(exprstr);
  PlanLexer lexer(&input);
  antlr4::CommonTokenStream tokens(&lexer);
  PlanParser parser(&tokens);

  PlanParser::ExprContext *tree = parser.expr();

  milvus::proto::schema::CollectionSchema schema;
  auto field = schema.add_fields();
  field->set_name("x");
  field->set_fieldid(100);
  field->set_data_type(milvus::proto::schema::DataType::Int64);

  field = schema.add_fields();
  field->set_name("v");
  field->set_fieldid(101);
  field->set_data_type(milvus::proto::schema::DataType::FloatVector);
  auto kv = field->add_type_params();
  kv->set_key("dim");
  kv->set_value("128");

  auto helper = milvus::CreateSchemaHelper(&schema);

  milvus::PlanCCVisitor visitor(&helper);
  auto res = std::any_cast<milvus::ExprWithDtype>(visitor.visit(tree));

  std::cout << res.expr->DebugString() << std::endl;
  return 0;
}
