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
        createBinExpr<proto::plan::BinaryExpr_BinaryOp_LogicalAnd>(
            left_expr, right_expr, this->arena.get()),
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
          return ExprWithDtype(
              createValueExpr<int64_t>(std::any_cast<int64_t>(left_value) *
                                           std::any_cast<int64_t>(right_value),
                                       this->arena.get()),
              proto::schema::DataType::Int64, false);
        case PlanParser::DIV:
          return ExprWithDtype(
              createValueExpr<int64_t>(std::any_cast<int64_t>(left_value) /
                                           std::any_cast<int64_t>(right_value),
                                       this->arena.get()),
              proto::schema::DataType::Int64, false);
        case PlanParser::MOD:
          return ExprWithDtype(
              createValueExpr<int64_t>(std::any_cast<int64_t>(left_value) %
                                           std::any_cast<int64_t>(right_value),
                                       this->arena.get()),
              proto::schema::DataType::Int64, false);
        default:
          assert(false);
        }
      }

      if (left_value.type() == typeid(double) &&
          right_value.type() == typeid(int64_t)) {
        switch (ctx->op->getType()) {
        case PlanParser::MUL:
          return ExprWithDtype(
              createValueExpr<double>(std::any_cast<double>(left_value) *
                                          std::any_cast<int64_t>(right_value),
                                      this->arena.get()),
              proto::schema::DataType::Double, false);
        case PlanParser::DIV:
          return ExprWithDtype(
              createValueExpr<double>(std::any_cast<double>(left_value) /
                                          std::any_cast<int64_t>(right_value),
                                      this->arena.get()),
              proto::schema::DataType::Double, false);
        default:
          assert(false);
        }
      }

      if (left_value.type() == typeid(int64_t) &&
          right_value.type() == typeid(double)) {

        switch (ctx->op->getType()) {
        case PlanParser::MUL:
          return ExprWithDtype(createValueExpr<double>(
                                   double(std::any_cast<int64_t>(left_value)) *
                                       std::any_cast<double>(right_value),
                                   this->arena.get()),
                               proto::schema::DataType::Double, false);
        case PlanParser::DIV:
          return ExprWithDtype(createValueExpr<double>(
                                   double(std::any_cast<int64_t>(left_value)) *
                                       std::any_cast<double>(right_value),
                                   this->arena.get()),
                               proto::schema::DataType::Double, false);
        default:
          assert(false);
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
    return nullptr;
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
      return ExprWithDtype(                                                    \
          createValueExpr<bool>(std::any_cast<left_type>(left_value) ==        \
                                    std::any_cast<right_type>(right_value),    \
                                this->arena.get()),                            \
          proto::schema::DataType::Bool, false);                               \
    case PlanParser::NE:                                                       \
      return ExprWithDtype(                                                    \
          createValueExpr<bool>(std::any_cast<left_type>(left_value) !=        \
                                    std::any_cast<right_type>(right_value),    \
                                this->arena.get()),                            \
          proto::schema::DataType::Bool, false);                               \
    }                                                                          \
  }

      PROCESS_EQALITY(bool, bool);
      PROCESS_EQALITY(std::string, std::string);
      PROCESS_EQALITY(double, double);
      PROCESS_EQALITY(double, int64_t);
      PROCESS_EQALITY(int64_t, double);
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

    if (left_expr_with_type.expr->has_value_expr() &&
        !right_expr_with_type.expr->has_value_expr()) {
      ExprWithDtype left = toValueExpr(
          google::protobuf::Arena::CreateMessage<proto::plan::GenericValue>(
              this->arena.get(),
              left_expr_with_type.expr->value_expr().value()),
          this->arena.get());
      ExprWithDtype right = right_expr_with_type;

      return ExprWithDtype(
          HandleCompare(ctx->op->getType(), left, right, this->arena.get()),
          proto::schema::DataType::Bool, false);
    }

    if (!left_expr_with_type.expr->has_value_expr() &&
        right_expr_with_type.expr->has_value_expr()) {

      ExprWithDtype left = left_expr_with_type;
      ExprWithDtype right = toValueExpr(
          google::protobuf::Arena::CreateMessage<proto::plan::GenericValue>(
              this->arena.get(),
              right_expr_with_type.expr->value_expr().value()));

      return ExprWithDtype(
          HandleCompare(ctx->op->getType(), left, right, this->arena.get()),
          proto::schema::DataType::Bool, false);
    }

    if (!left_expr_with_type.expr->has_value_expr() &&
        !right_expr_with_type.expr->has_value_expr()) {

      return ExprWithDtype(
          HandleCompare(ctx->op->getType(), left_expr_with_type,
                        right_expr_with_type, this->arena.get()),
          proto::schema::DataType::Bool, false);
    }

    return nullptr;
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

    std::vector<std::string> nested_path;
    auto field = helper->GetFieldFromNameDefaultJSON(fieldname);
    assert(field.data_type() == proto::schema::DataType::JSON ||
           field.data_type() == proto::schema::DataType::Array);
    if (fieldname != field.name())
      nested_path.push_back(fieldname);
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
        path_ = path_.substr(1, path_.length() - 2);
        assert(path_ != "");
      }
      nested_path.push_back(path_);
    }

    auto info = google::protobuf::Arena::CreateMessage<proto::plan::ColumnInfo>(
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

  virtual std::any
  visitReverseRange(PlanParser::ReverseRangeContext *ctx) override {
    auto info = getChildColumnInfo(ctx->Identifier(), ctx->JSONIdentifier());
    assert(info != nullptr);
    assert(checkDirectComparisonBinaryField(info));
    auto lower = std::any_cast<ExprWithDtype>(ctx->expr()[1]->accept(this));
    auto upper = std::any_cast<ExprWithDtype>(ctx->expr()[0]->accept(this));

    if (info->data_type() == proto::schema::DataType::Int8 ||
        info->data_type() == proto::schema::DataType::Int16 ||
        info->data_type() == proto::schema::DataType::Int32 ||
        info->data_type() == proto::schema::DataType::Int64 ||
        info->data_type() == proto::schema::DataType::Float ||
        info->data_type() == proto::schema::DataType::Double ||
        info->data_type() == proto::schema::DataType::String) {
      auto a = extractValue(lower.expr);
      auto b = extractValue(upper.expr);
      if (a.has_value() && b.has_value()) {
        bool lowerinclusive = ctx->op1->getType() == PlanParser::GE;
        bool upperinclusive = ctx->op2->getType() == PlanParser::GE;
        auto expr = new proto::plan::Expr();
        auto binary_range_expr = new proto::plan::BinaryRangeExpr();
        auto lower_value = new proto::plan::GenericValue();
        auto upper_value = new proto::plan::GenericValue();
        if (a.type() == typeid(int8_t))
          lower_value->set_int64_val(int64_t(std::any_cast<int8_t>(a)));
        if (a.type() == typeid(int16_t))
          lower_value->set_int64_val(int64_t(std::any_cast<int16_t>(a)));
        if (a.type() == typeid(int32_t))
          lower_value->set_int64_val(int64_t(std::any_cast<int32_t>(a)));
        if (a.type() == typeid(int64_t))
          lower_value->set_int64_val(std::any_cast<int64_t>(a));
        if (a.type() == typeid(double))
          lower_value->set_float_val(std::any_cast<double>(a));
        if (a.type() == typeid(float))
          lower_value->set_float_val(double(std::any_cast<float>(a)));
        if (a.type() == typeid(std::string))
          lower_value->set_string_val(std::any_cast<std::string>(a));

        if (b.type() == typeid(int8_t))
          upper_value->set_int64_val(int64_t(std::any_cast<int8_t>(b)));
        if (b.type() == typeid(int16_t))
          upper_value->set_int64_val(int64_t(std::any_cast<int16_t>(b)));
        if (b.type() == typeid(int32_t))
          upper_value->set_int64_val(int64_t(std::any_cast<int32_t>(b)));
        if (b.type() == typeid(int64_t))
          upper_value->set_int64_val(int64_t(std::any_cast<int64_t>(b)));
        if (b.type() == typeid(double))
          upper_value->set_float_val(std::any_cast<double>(b));
        if (b.type() == typeid(float))
          upper_value->set_float_val(double(std::any_cast<float>(b)));
        if (b.type() == typeid(std::string))
          upper_value->set_string_val(std::any_cast<std::string>(b));

        binary_range_expr->set_lower_inclusive(lowerinclusive);
        binary_range_expr->set_upper_inclusive(upperinclusive);
        binary_range_expr->unsafe_arena_set_allocated_column_info(info);

        binary_range_expr->unsafe_arena_set_allocated_lower_value(lower_value);
        binary_range_expr->unsafe_arena_set_allocated_upper_value(upper_value);
        expr->set_allocated_binary_range_expr(binary_range_expr);
        return ExprWithDtype(expr, proto::schema::DataType::Bool, false);
      }
    }

    return nullptr;
  }

  virtual std::any visitAddSub(PlanParser::AddSubContext *ctx) override {
    auto left_expr_with_type =
        std::any_cast<ExprWithDtype>(ctx->expr()[0]->accept(this));
    auto right_expr_with_type =
        std::any_cast<ExprWithDtype>(ctx->expr()[1]->accept(this));
    auto left_value = extractValue(left_expr_with_type.expr);
    auto right_value = extractValue(right_expr_with_type.expr);

    if (left_value.has_value() && right_value.has_value()) {

#define PROCESS_ADDSUB(left_type, right_type, target_type, datatype)           \
  if (left_value.type() == typeid(left_type) &&                                \
      right_value.type() == typeid(right_type)) {                              \
    switch (ctx->op->getType()) {                                              \
    case PlanParser::ADD:                                                      \
      return ExprWithDtype(createValueExpr<target_type>(                       \
                               std::any_cast<left_type>(left_value) +          \
                                   std::any_cast<right_type>(right_value),     \
                               this->arena.get()),                             \
                           datatype, false);                                   \
    case PlanParser::SUB:                                                      \
      return ExprWithDtype(createValueExpr<target_type>(                       \
                               std::any_cast<left_type>(left_value) -          \
                                   std::any_cast<right_type>(right_value),     \
                               this->arena.get()),                             \
                           datatype, false);                                   \
    default:                                                                   \
      assert(false);                                                           \
    }                                                                          \
  }

      PROCESS_ADDSUB(double, double, double, proto::schema::DataType::Double)
      PROCESS_ADDSUB(double, int64_t, double, proto::schema::DataType::Double)
      PROCESS_ADDSUB(int64_t, double, double, proto::schema::DataType::Double)
      PROCESS_ADDSUB(int64_t, int64_t, int64_t, proto::schema::DataType::Int64)
      PROCESS_ADDSUB(int32_t, int32_t, int32_t, proto::schema::DataType::Int32)
      PROCESS_ADDSUB(float, float, double, proto::schema::DataType::Double)
    }

    auto left_expr = left_expr_with_type.expr;
    auto right_expr = right_expr_with_type.expr;
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
      else if (arithmeticDtype(right_expr_with_type.dtype))
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
    case PlanParser::ADD:
      return ExprWithDtype(
          createBinArithExpr<proto::plan::ArithOpType::Add>(
              left_expr, right_expr, this->arena.get()),
          calDataType(&left_expr_with_type, &right_expr_with_type), false);
    case PlanParser::SUB:
      return ExprWithDtype(
          createBinArithExpr<proto::plan::ArithOpType::Sub>(
              left_expr, right_expr, this->arena.get()),
          calDataType(&left_expr_with_type, &right_expr_with_type), false);

    default:
      assert(false);
    }
  }

  virtual std::any
  visitRelational(PlanParser::RelationalContext *ctx) override {
    auto left_expr_with_type =
        std::any_cast<ExprWithDtype>(ctx->expr()[0]->accept(this));
    auto right_expr_with_type =
        std::any_cast<ExprWithDtype>(ctx->expr()[1]->accept(this));
    auto left_value = extractValue(left_expr_with_type.expr);
    auto right_value = extractValue(right_expr_with_type.expr);
    if (left_value.has_value() && right_value.has_value()) {

#define PROCESS_RELATIONAL(left_type, right_type)                              \
  if (left_value.type() == typeid(left_type) &&                                \
      right_value.type() == typeid(right_type)) {                              \
                                                                               \
    switch (ctx->op->getType()) {                                              \
    case PlanParser::LT:                                                       \
      return ExprWithDtype(                                                    \
          createValueExpr<bool>(std::any_cast<left_type>(left_value) <         \
                                    std::any_cast<right_type>(right_value),    \
                                this->arena.get()),                            \
          proto::schema::DataType::Bool, false);                               \
    case PlanParser::LE:                                                       \
      return ExprWithDtype(                                                    \
          createValueExpr<bool>(std::any_cast<left_type>(left_value) <=        \
                                    std::any_cast<right_type>(right_value),    \
                                this->arena.get()),                            \
          proto::schema::DataType::Bool, false);                               \
    case PlanParser::GT:                                                       \
      return ExprWithDtype(                                                    \
          createValueExpr<bool>(std::any_cast<left_type>(left_value) >         \
                                    std::any_cast<right_type>(right_value),    \
                                this->arena.get()),                            \
          proto::schema::DataType::Bool, false);                               \
    case PlanParser::GE:                                                       \
      return ExprWithDtype(                                                    \
          createValueExpr<bool>(std::any_cast<left_type>(left_value) >=        \
                                    std::any_cast<right_type>(right_value),    \
                                this->arena.get()),                            \
          proto::schema::DataType::Bool, false);                               \
    default:                                                                   \
      assert(false);                                                           \
    }                                                                          \
  }

      PROCESS_RELATIONAL(double, double)
      PROCESS_RELATIONAL(double, int64_t)
      PROCESS_RELATIONAL(int64_t, double)
      PROCESS_RELATIONAL(std::string, std::string)
      PROCESS_RELATIONAL(int64_t, int64_t)
      PROCESS_RELATIONAL(int32_t, int32_t)
    }

    if (left_expr_with_type.expr->has_value_expr() &&
        !right_expr_with_type.expr->has_value_expr()) {
      ExprWithDtype left = toValueExpr(
          google::protobuf::Arena::CreateMessage<proto::plan::GenericValue>(
              this->arena.get(),
              left_expr_with_type.expr->value_expr().value()));
      ExprWithDtype right = right_expr_with_type;

      return ExprWithDtype(
          HandleCompare(ctx->op->getType(), left, right, this->arena.get()),
          proto::schema::DataType::Bool, false);
    }

    if (!left_expr_with_type.expr->has_value_expr() &&
        right_expr_with_type.expr->has_value_expr()) {
      ExprWithDtype left = left_expr_with_type;
      ExprWithDtype right = toValueExpr(
          google::protobuf::Arena::CreateMessage<proto::plan::GenericValue>(
              this->arena.get(),
              right_expr_with_type.expr->value_expr().value()));

      return ExprWithDtype(
          HandleCompare(ctx->op->getType(), left, right, this->arena.get()),
          proto::schema::DataType::Bool, false);
    }

    if (!left_expr_with_type.expr->has_value_expr() &&
        !right_expr_with_type.expr->has_value_expr()) {

      return ExprWithDtype(
          HandleCompare(ctx->op->getType(), left_expr_with_type,
                        right_expr_with_type, this->arena.get()),
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
    auto expr = google::protobuf::Arena::CreateMessage<proto::plan::Expr>(
        this->arena.get());
    auto bin_arith_expr =
        google::protobuf::Arena::CreateMessage<proto::plan::BinaryArithExpr>(
            this->arena.get());
    auto column_expr =
        google::protobuf::Arena::CreateMessage<proto::plan::ColumnExpr>(
            this->arena.get());
    column_expr->unsafe_arena_set_allocated_info(info);
    auto left_expr = google::protobuf::Arena::CreateMessage<proto::plan::Expr>(
        this->arena.get());
    left_expr->unsafe_arena_set_allocated_column_expr(column_expr);
    bin_arith_expr->unsafe_arena_set_allocated_left(left_expr);
    bin_arith_expr->set_op(proto::plan::ArithOpType::ArrayLength);
    expr->unsafe_arena_set_allocated_binary_arith_expr(bin_arith_expr);
    return ExprWithDtype(expr, proto::schema::DataType::Int64, false);
  }

  virtual std::any visitTerm(PlanParser::TermContext *ctx) override {
    auto first_expr_with_type =
        std::any_cast<ExprWithDtype>(ctx->expr()[0]->accept(this));
    auto info = first_expr_with_type.expr->unsafe_arena_release_column_expr()
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
      auto value = extractValue(expr_);
      if (value.type() == typeid(int8_t)) {
        v->set_int64_val(int64_t(std::any_cast<int8_t>(value)));
        term_expr->mutable_values()->UnsafeArenaAddAllocated(v);
        continue;
      }

      if (value.type() == typeid(int64_t)) {
        v->set_int64_val(int64_t(std::any_cast<int64_t>(value)));
        term_expr->mutable_values()->UnsafeArenaAddAllocated(v);
        continue;
      }

      if (value.type() == typeid(int32_t)) {
        v->set_int64_val(int64_t(std::any_cast<int32_t>(value)));
        term_expr->mutable_values()->UnsafeArenaAddAllocated(v);
        continue;
      }

      if (value.type() == typeid(double)) {
        v->set_float_val(double(std::any_cast<double>(value)));
        term_expr->mutable_values()->UnsafeArenaAddAllocated(v);
        continue;
      }

      if (value.type() == typeid(float)) {
        v->set_float_val(double(std::any_cast<float>(value)));
        term_expr->mutable_values()->UnsafeArenaAddAllocated(v);
        continue;
      }

      if (value.type() == typeid(bool)) {
        v->set_bool_val(std::any_cast<bool>(value));
        term_expr->mutable_values()->UnsafeArenaAddAllocated(v);
        continue;
      }

      if (value.type() == typeid(std::string)) {
        v->set_string_val(std::any_cast<std::string>(value));
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

    auto expr = google::protobuf::Arena::CreateMessage<proto::plan::Expr>(
        this->arena.get());
    auto json_contain_expr =
        google::protobuf::Arena::CreateMessage<proto::plan::JSONContainsExpr>(
            this->arena.get());
    auto value = json_contain_expr->add_elements();
    value->unsafe_arena_set_allocated_array_val(
        google::protobuf::Arena::CreateMessage<proto::plan::Array>(
            this->arena.get(), elem.expr->value_expr().value().array_val()));
    json_contain_expr->set_elements_same_type(true);
    json_contain_expr->set_allocated_column_info(
        new proto::plan::ColumnInfo(info));
    json_contain_expr->unsafe_arena_set_allocated_column_info(
        google::protobuf::Arena::CreateMessage<proto::plan::ColumnInfo>(
            this->arena.get(), info));
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

    if (info->data_type() == proto::schema::DataType::Int8 ||
        info->data_type() == proto::schema::DataType::Int16 ||
        info->data_type() == proto::schema::DataType::Int32 ||
        info->data_type() == proto::schema::DataType::Int64 ||
        info->data_type() == proto::schema::DataType::Float ||
        info->data_type() == proto::schema::DataType::Double ||
        info->data_type() == proto::schema::DataType::String) {
      auto a = extractValue(lower.expr);
      auto b = extractValue(upper.expr);
      if (a.has_value() && b.has_value()) {
        bool lowerinclusive = ctx->op1->getType() == PlanParser::LE;
        bool upperinclusive = ctx->op2->getType() == PlanParser::LE;
        auto expr = new proto::plan::Expr();
        auto binary_range_expr = new proto::plan::BinaryRangeExpr();
        auto lower_value = new proto::plan::GenericValue();
        auto upper_value = new proto::plan::GenericValue();
        if (a.type() == typeid(int8_t))
          lower_value->set_int64_val(int64_t(std::any_cast<int8_t>(a)));
        if (a.type() == typeid(int16_t))
          lower_value->set_int64_val(int64_t(std::any_cast<int16_t>(a)));
        if (a.type() == typeid(int32_t))
          lower_value->set_int64_val(int64_t(std::any_cast<int32_t>(a)));
        if (a.type() == typeid(int64_t))
          lower_value->set_int64_val(std::any_cast<int64_t>(a));
        if (a.type() == typeid(double))
          lower_value->set_float_val(std::any_cast<double>(a));
        if (a.type() == typeid(float))
          lower_value->set_float_val(double(std::any_cast<float>(a)));
        if (a.type() == typeid(std::string))
          lower_value->set_string_val(std::any_cast<std::string>(a));

        if (b.type() == typeid(int8_t))
          upper_value->set_int64_val(int64_t(std::any_cast<int8_t>(b)));
        if (b.type() == typeid(int16_t))
          upper_value->set_int64_val(int64_t(std::any_cast<int16_t>(b)));
        if (b.type() == typeid(int32_t))
          upper_value->set_int64_val(int64_t(std::any_cast<int32_t>(b)));
        if (b.type() == typeid(int64_t))
          upper_value->set_int64_val(int64_t(std::any_cast<int64_t>(b)));
        if (b.type() == typeid(double))
          upper_value->set_float_val(std::any_cast<double>(b));
        if (b.type() == typeid(float))
          upper_value->set_float_val(double(std::any_cast<float>(b)));
        if (b.type() == typeid(std::string))
          upper_value->set_string_val(std::any_cast<std::string>(b));

        binary_range_expr->set_lower_inclusive(lowerinclusive);
        binary_range_expr->set_upper_inclusive(upperinclusive);
        binary_range_expr->unsafe_arena_set_allocated_column_info(info);

        binary_range_expr->unsafe_arena_set_allocated_lower_value(lower_value);
        binary_range_expr->unsafe_arena_set_allocated_upper_value(upper_value);
        expr->set_allocated_binary_range_expr(binary_range_expr);
        return ExprWithDtype(expr, proto::schema::DataType::Bool, false);
      }
    }

    return nullptr;
  }

  virtual std::any visitUnary(PlanParser::UnaryContext *ctx) override {
    auto expr_with_dtype =
        std::any_cast<ExprWithDtype>(ctx->expr()->accept(this));
    auto value = extractValue(expr_with_dtype.expr);
    if (value.has_value()) {

#define PROCESS_UNARY(dtype, schema_dtype)                                     \
  if (value.type() == typeid(dtype)) {                                         \
    switch (ctx->op->getType()) {                                              \
    case PlanParser::ADD:                                                      \
      return expr_with_dtype;                                                  \
    case PlanParser::SUB:                                                      \
      return ExprWithDtype(                                                    \
          createValueExpr<dtype>(-std::any_cast<dtype>(value),                 \
                                 this->arena.get()),                           \
          schema_dtype, false);                                                \
    case PlanParser::NOT:                                                      \
      return ExprWithDtype(                                                    \
          createValueExpr<dtype>(!std::any_cast<dtype>(value),                 \
                                 this->arena.get()),                           \
          proto::schema::DataType::Bool, false);                               \
    default:                                                                   \
      assert(false);                                                           \
    }                                                                          \
  }

      PROCESS_UNARY(double, proto::schema::DataType::Float);
      PROCESS_UNARY(float, proto::schema::DataType::Float);
      PROCESS_UNARY(int8_t, proto::schema::DataType::Int64);
      PROCESS_UNARY(int32_t, proto::schema::DataType::Int64);
      PROCESS_UNARY(int64_t, proto::schema::DataType::Int64);
      PROCESS_UNARY(bool, proto::schema::DataType::Bool);
    }

    assert(checkDirectComparisonBinaryField(
        google::protobuf::Arena::CreateMessage<proto::plan::ColumnInfo>(
            this->arena.get(), expr_with_dtype.expr->column_expr().info())));

    switch (ctx->op->getType()) {
    case PlanParser::ADD:
      return expr_with_dtype.expr;
    case PlanParser::NOT:
      assert(!expr_with_dtype.dependent &&
             expr_with_dtype.dtype == proto::schema::DataType::Bool);
      auto expr = google::protobuf::Arena::CreateMessage<proto::plan::Expr>(
          this->arena.get());
      auto unary_expr =
          google::protobuf::Arena::CreateMessage<proto::plan::UnaryExpr>(
              this->arena.get());
      unary_expr->unsafe_arena_set_allocated_child(expr_with_dtype.expr);
      unary_expr->set_op(proto::plan::UnaryExpr_UnaryOp_Not);
      return ExprWithDtype(expr, proto::schema::Bool, false);
    }
    return nullptr;
  }

  virtual std::any visitArray(PlanParser::ArrayContext *ctx) override {

    auto expr = google::protobuf::Arena::CreateMessage<proto::plan::Expr>(
        this->arena.get());
    auto array_expr =
        google::protobuf::Arena::CreateMessage<proto::plan::Array>(
            this->arena.get());
    auto dtype = proto::schema::DataType::None;

    auto is_same = true;
    for (auto &&elem : ctx->expr()) {
      auto expr_ = std::any_cast<ExprWithDtype>(elem->accept(this)).expr;
      auto v = array_expr->add_array();
      auto value = extractValue(expr_);
      if (value.has_value()) {
        if (value.type() == typeid(int8_t)) {
          v->set_int64_val(int64_t(std::any_cast<int8_t>(value)));
          if (dtype != proto::schema::DataType::None &&
              dtype != proto::schema::DataType::Int8) {
            is_same = false;
          }
          if (dtype == proto::schema::DataType::None) {
            dtype = proto::schema::DataType::Int8;
          }
          continue;
        }
        if (value.type() == typeid(int16_t)) {
          v->set_int64_val(int64_t(std::any_cast<int16_t>(value)));
          if (dtype != proto::schema::DataType::None &&
              dtype != proto::schema::DataType::Int16) {
            is_same = false;
          }
          if (dtype == proto::schema::DataType::None) {
            dtype = proto::schema::DataType::Int16;
          }
          continue;
        }
        if (value.type() == typeid(int32_t)) {
          v->set_int64_val(int64_t(std::any_cast<int32_t>(value)));
          if (dtype != proto::schema::DataType::None &&
              dtype != proto::schema::DataType::Int32) {
            is_same = false;
          }
          if (dtype == proto::schema::DataType::None) {

            dtype = proto::schema::DataType::Int32;
          }
          continue;
        }

        if (value.type() == typeid(int64_t)) {
          v->set_int64_val(std::any_cast<int64_t>(value));
          if (dtype != proto::schema::DataType::None &&
              dtype != proto::schema::DataType::Int64) {
            is_same = false;
          }
          if (dtype == proto::schema::DataType::None) {
            dtype = proto::schema::DataType::Int64;
          }
          continue;
        }

        if (value.type() == typeid(double)) {
          v->set_float_val(std::any_cast<double>(value));
          if (dtype != proto::schema::DataType::None &&
              dtype != proto::schema::DataType::Double) {
            is_same = false;
          }
          if (dtype == proto::schema::DataType::None) {
            dtype = proto::schema::DataType::Double;
          }
          continue;
        }

        if (value.type() == typeid(float)) {
          v->set_float_val(std::any_cast<float>(value));
          if (dtype != proto::schema::DataType::None &&
              dtype != proto::schema::DataType::Float) {
            is_same = false;
          }
          if (dtype == proto::schema::DataType::None) {
            dtype = proto::schema::DataType::Float;
          }
          continue;
        }
      }
    }

    auto generic_value =
        google::protobuf::Arena::CreateMessage<proto::plan::GenericValue>(
            this->arena.get());

    generic_value->unsafe_arena_set_allocated_array_val(array_expr);

    auto value_expr =
        google::protobuf::Arena::CreateMessage<proto::plan::ValueExpr>(
            this->arena.get());
    value_expr->unsafe_arena_set_allocated_value(generic_value);
    expr->unsafe_arena_set_allocated_value_expr(value_expr);
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

    auto expr = google::protobuf::Arena::CreateMessage<proto::plan::Expr>(
        this->arena.get());
    auto json_contain_expr =
        google::protobuf::Arena::CreateMessage<proto::plan::JSONContainsExpr>(
            this->arena.get());

    auto value = json_contain_expr->add_elements();
    value->unsafe_arena_set_allocated_array_val(
        google::protobuf::Arena::CreateMessage<proto::plan::Array>(
            this->arena.get(), elem.expr->value_expr().value().array_val()));

    json_contain_expr->set_elements_same_type(
        elem.expr->value_expr().value().array_val().same_type());
    json_contain_expr->unsafe_arena_set_allocated_column_info(
        google::protobuf::Arena::CreateMessage<proto::plan::ColumnInfo>(
            this->arena.get(), info));
    json_contain_expr->set_op(proto::plan::JSONContainsExpr_JSONOp_ContainsAny);
    expr->unsafe_arena_set_allocated_json_contains_expr(json_contain_expr);
    return ExprWithDtype(expr, proto::schema::Bool, false);
  }

  virtual std::any visitExists(PlanParser::ExistsContext *ctx) override {
    auto a = std::any_cast<ExprWithDtype>(ctx->expr());
    auto info = a.expr->column_expr().info();
    assert(info.data_type() == proto::schema::DataType::Array);
    auto expr = google::protobuf::Arena::CreateMessage<proto::plan::Expr>(
        this->arena.get());

    auto col_expr =
        google::protobuf::Arena::CreateMessage<proto::plan::ColumnExpr>(
            this->arena.get());
    col_expr->unsafe_arena_set_allocated_info(
        new proto::plan::ColumnInfo(info));
    col_expr->unsafe_arena_set_allocated_info(
        google::protobuf::Arena::CreateMessage<proto::plan::ColumnInfo>(
            this->arena.get(), info));
    expr->unsafe_arena_set_allocated_column_expr(col_expr);
    return ExprWithDtype(expr, proto::schema::DataType::Bool, false);
  }

  virtual std::any visitEmptyTerm(PlanParser::EmptyTermContext *ctx) override {

    auto first = std::any_cast<ExprWithDtype>(ctx->expr()->accept(this));
    auto info = first.expr->column_expr().info();

    auto expr = google::protobuf::Arena::CreateMessage<proto::plan::Expr>(
        this->arena.get());
    auto col_expr =
        google::protobuf::Arena::CreateMessage<proto::plan::ColumnExpr>(
            this->arena.get());
    auto term_expr =
        google::protobuf::Arena::CreateMessage<proto::plan::TermExpr>(
            this->arena.get());

    expr->unsafe_arena_set_allocated_term_expr(term_expr);
    col_expr->unsafe_arena_set_allocated_info(
        google::protobuf::Arena::CreateMessage<proto::plan::ColumnInfo>(
            this->arena.get(), info));
    expr->unsafe_arena_set_allocated_column_expr(col_expr);
    expr->unsafe_arena_set_allocated_term_expr(term_expr);
    return ExprWithDtype(expr, proto::schema::DataType::Bool, false);
  }

  PlanCCVisitor(SchemaHelper *const helper)
      : helper(helper), arena(std::make_shared<google::protobuf::Arena>()) {}

private:
  SchemaHelper *helper;
  std::shared_ptr<google::protobuf::Arena> arena;
};
} // namespace milvus

void check_expr(const std::string &exprstr, milvus::SchemaHelper &helper) {

  antlr4::ANTLRInputStream input(exprstr);
  PlanLexer lexer(&input);
  antlr4::CommonTokenStream tokens(&lexer);
  PlanParser parser(&tokens);
  PlanParser::ExprContext *tree = parser.expr();
  milvus::PlanCCVisitor visitor(&helper);
  auto res = std::any_cast<milvus::ExprWithDtype>(visitor.visit(tree));
  std::cout << exprstr << "\t" << res.expr->DebugString() << std::endl;
}

int main() {

  milvus::proto::schema::CollectionSchema schema;
  {
    auto field = schema.add_fields();
    field->set_name("x_int64");
    field->set_fieldid(100);
    field->set_data_type(milvus::proto::schema::DataType::Int64);
  }
  {
    auto field = schema.add_fields();
    field->set_name("x_int32");
    field->set_fieldid(101);
    field->set_data_type(milvus::proto::schema::DataType::Int32);
  }
  {
    auto field = schema.add_fields();
    field->set_name("x_int16");
    field->set_fieldid(102);
    field->set_data_type(milvus::proto::schema::DataType::Int16);
  }
  {
    auto field = schema.add_fields();
    field->set_name("x_int8");
    field->set_fieldid(103);
    field->set_data_type(milvus::proto::schema::DataType::Int8);
  }
  {
    auto field = schema.add_fields();
    field->set_name("x_bool");
    field->set_fieldid(104);
    field->set_data_type(milvus::proto::schema::Bool);
  }
  {
    auto field = schema.add_fields();
    field->set_name("x_float");
    field->set_fieldid(105);
    field->set_data_type(milvus::proto::schema::DataType::Float);
  }
  {
    auto field = schema.add_fields();
    field->set_name("x_double");
    field->set_fieldid(106);
    field->set_data_type(milvus::proto::schema::DataType::Double);
  }
  {
    auto field = schema.add_fields();
    field->set_name("x_string");
    field->set_fieldid(107);
    field->set_data_type(milvus::proto::schema::DataType::String);
  }
  {
    auto field = schema.add_fields();
    field->set_name("v");
    field->set_fieldid(108);
    field->set_data_type(milvus::proto::schema::DataType::FloatVector);
    auto kv = field->add_type_params();
    kv->set_key("dim");
    kv->set_value("128");
  }
  auto helper = milvus::CreateSchemaHelper(&schema);
  /*
  const std::vector<std::string> for_test = {

      "1.0 + 2.0",
      "1.0 + 2",
      "1 + 2.0",
      "1 + 2",
      "1.0 - 2.0",
      "1.0 - 2",
      "1 - 2.0",
      "1 - 2",
      "1.0 * 2.0",
      "1.0 * 2",
      "1 * 2.0",
      "1 * 2",
      "1.0 / 2.0",
      "1.0 / 2",
      "1 / 2.0",
      "1 / 2",
      "1 % 2",
      // ------------------- logical operations ----------------
      "true and false",
      "true or false",
      "!true",
      "!false",
      // ------------------- relational operations ----------------
      "\"1\" < \"2\"",
      "1.0 < 2.0",
      "1.0 < 2",
      "1 < 2.0",
      "1 < 2",
      "\" 1 \" <= \" 2 \"",
      "1.0 <= 2.0",
      "1.0 <= 2",
      "1 <= 2.0",
      "1 <= 2",
      "\" 1 \" > \" 2 \"",
      "1.0 > 2.0",
      "1.0 > 2",
      "1 > 2.0",
      "1 > 2",
      "\" 1 \" >= \" 2 \"",
      "1.0 >= 2.0",
      "1.0 >= 2",
      "1 >= 2.0",
      "1 >= 2",
      "\" 1 \" == \" 2 \"",
      "1.0 == 2.0",
      "1.0 == 2",
      "1 == 2.0",
      "1 == 2",
      "true == false",
      "\" 1 \" != \" 2 \"",
      "1.0 != 2.0",
      "1.0 != 2",
      "1 != 2.0",
      "1 != 2",
      "true != false"
  };
  */

  std::vector<std::string> for_test = {

      "x_int64 > 30",
      "x_int64 in [4,5,6]",
      "100<x_int32<120",
      "32<=x_int16<64",
      "12<=x_int8<=32",
      "12<x_int64<=22",
      "1.0<x_float<=3.0",
      "2.0<=x_double<=10.0",
      "x_bool in [true]",
      "x_string in [\"aa\", \"bb\"]",
      "x_int64 in [1, \"abc\", 2]",
      "x_int64 >= x_int32",
      "x_int64 <x_int32",
      "x_double==x_float",
      "x_double!=x_float",
      "x_int64%10==9",
      "x_int64-10==9",
      "x_int64*3==9",
      "x_int64/10==3",
      "x_int64+1==3",
      "100>x_int32>120",
      "64>=x_int16>32",
      "12>x_int8>=10",
      "12>=x_int64>6",
      "3.0>x_float>1.0",
      "15.0>x_double>10.0",
      R"("str14" > x_string > "str13")",
      "1",
      "2.0",
      "true",
      "false",
      R"("str")",
      "x_int64 > 30 and x_bool",
  };

  for (auto &&str_ : for_test) {
    std::cout << str_ << std::endl;
    check_expr(str_, helper);
  }

  return 0;
}
